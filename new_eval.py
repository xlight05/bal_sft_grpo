# %%
%%capture
import os
if "COLAB_" not in "".join(os.environ.keys()):
    !pip install unsloth
else:
    # Do this only in Colab notebooks! Otherwise use pip install unsloth
    !pip install --no-deps bitsandbytes accelerate xformers==0.0.29.post3 peft trl==0.15.2 triton cut_cross_entropy unsloth_zoo
    !pip install sentencepiece protobuf "datasets>=3.4.1" huggingface_hub hf_transfer
    !pip install transformers==4.51.3
    !pip install --no-deps unsloth
    !pip install nltk rouge-score numpy requests
    !pip install ballerina-platform-codebleu
    !pip install tree-sitter-ballerina

# %%
SYSTEM_PROMPT = """You are a pragmatic Ballerina programmer who enjoys test driven development. Given the following question, write a Ballerina function to complete the task and then write the the unit tests to validate the function.

1. Make the code simple and easy to understand.
2. Try to limit library usage to the standard library. Be careful with your types, and try to limit yourself to the basic built in types and standard library functions.
3. Before you start writing the function you can think through how to solve the problem and perform reasoning in the comments above the function.
4. Then write unit tests for the function you defined. Make sure to write at least 4 assertions to test the function. The tests should be a simple.

Strictly follow the following output format for each response: Make sure to include code inside <CODE> and <TESTS> blocks.

# Overview
Brief overview about the solution.

<CODE>
```ballerina
// Reasoning goes here
// and can be multi-line
function add(int a, int b) returns int {
    return a + b;
}
```
</CODE>

<TESTS>
```ballerina
import ballerina/test;

@test:Config { }
function testAssertEquals() {
    int addResult = add(40, 2);
    test:assertEquals(addResult, 42);

    addResult = add(0, 0);
    test:assertEquals(addResult, 0);

    addResult = add(-1, 1);
    test:assertEquals(addResult, 0);

    addResult = add(-5, -5);
    test:assertEquals(addResult, -10);
}
```
</TESTS>

"""


# %%
# %%
"""
Ballerina Base Model Evaluator
Evaluates the base Qwen2.5-Coder-7B-Instruct model on Ballerina code generation tasks
before fine-tuning to establish baseline metrics.
"""

import torch
import json
import requests
import subprocess
import tempfile
import os
import re
from collections import defaultdict
from datasets import Dataset
from typing import List, Dict, Any
import numpy as np
from pathlib import Path
from datetime import datetime

# Install required packages
try:
    import nltk
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    nltk.download('punkt', quiet=True)
except ImportError:
    subprocess.run(["pip", "install", "nltk"], check=True)
    import nltk
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    nltk.download('punkt', quiet=True)

try:
    from rouge_score import rouge_scorer
except ImportError:
    subprocess.run(["pip", "install", "rouge-score"], check=True)
    from rouge_score import rouge_scorer

try:
    from codebleu import calc_codebleu, AVAILABLE_LANGS
except ImportError:
    subprocess.run(["pip", "install", "ballerina-platform-codebleu"], check=True)
    subprocess.run(["pip", "install", "tree-sitter-ballerina"], check=True)
    from codebleu import calc_codebleu, AVAILABLE_LANGS

try:
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template
except ImportError:
    subprocess.run(["pip", "install", "unsloth[colab-new]"], check=True)
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template

import difflib

import subprocess
import os
from typing import Dict, List, Optional, Tuple


class BallerinaManager:
    def __init__(self, project_path: str = "."):
        self.project_path = project_path

    def get_build_status(self) -> Dict[str, any]:
        try:
            result = subprocess.run(
                ["bal", "build", "--offline"],
                cwd=self.project_path,
                capture_output=True,
                text=True,
                timeout=60
            )

            return {
                "success": result.returncode == 0,
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "compilation_errors": self._extract_compilation_errors(result.stdout + result.stderr)
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "return_code": -1,
                "stdout": "",
                "stderr": "Build process timed out",
                "compilation_errors": ["Build process timed out after 60 seconds"]
            }
        except FileNotFoundError:
            return {
                "success": False,
                "return_code": -1,
                "stdout": "",
                "stderr": "bal command not found",
                "compilation_errors": ["Ballerina CLI not found. Please ensure Ballerina is installed and in PATH"]
            }
        except Exception as e:
            return {
                "success": False,
                "return_code": -1,
                "stdout": "",
                "stderr": str(e),
                "compilation_errors": [f"Unexpected error: {str(e)}"]
            }

    def get_test_status(self) -> Dict[str, any]:
        try:
            result = subprocess.run(
                ["bal", "test", "--offline"],
                cwd=self.project_path,
                capture_output=True,
                text=True,
                timeout=120
            )

            return {
                "success": result.returncode == 0,
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "test_results": self._extract_test_results(result.stdout + result.stderr),
                "compilation_errors": self._extract_compilation_errors(result.stdout + result.stderr)
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "return_code": -1,
                "stdout": "",
                "stderr": "Test process timed out",
                "test_results": {"passed": 0, "failed": 0, "total": 0},
                "compilation_errors": ["Test process timed out after 120 seconds"]
            }
        except FileNotFoundError:
            return {
                "success": False,
                "return_code": -1,
                "stdout": "",
                "stderr": "bal command not found",
                "test_results": {"passed": 0, "failed": 0, "total": 0},
                "compilation_errors": ["Ballerina CLI not found. Please ensure Ballerina is installed and in PATH"]
            }
        except Exception as e:
            return {
                "success": False,
                "return_code": -1,
                "stdout": "",
                "stderr": str(e),
                "test_results": {"passed": 0, "failed": 0, "total": 0},
                "compilation_errors": [f"Unexpected error: {str(e)}"]
            }

    def _extract_compilation_errors(self, output: str) -> List[str]:
        errors = []
        lines = output.split('\n')

        for line in lines:
            line = line.strip()
            if any(keyword in line.lower() for keyword in ['error:', 'compilation error', 'build failed']):
                errors.append(line)
            elif line.startswith('ERROR') or 'error occurred' in line.lower():
                errors.append(line)

        return errors

    def _extract_test_results(self, output: str) -> Dict[str, int]:
        results = {"passed": 0, "failed": 0, "total": 0}
        lines = output.split('\n')

        for line in lines:
            line = line.strip()

            # Look for Ballerina test output format: "X passing", "Y failing", "Z skipped"
            if 'passing' in line:
                try:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == 'passing' and i > 0:
                            results["passed"] = int(parts[i-1])
                            break
                except (ValueError, IndexError):
                    continue
            elif 'failing' in line:
                try:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == 'failing' and i > 0:
                            results["failed"] = int(parts[i-1])
                            break
                except (ValueError, IndexError):
                    continue

        results["total"] = results["passed"] + results["failed"]
        return results



from uuid import uuid4
import tempfile
from pathlib import Path

"""
Define functions for setting up and testing Ballerina projects.
"""

def create_ballerina_toml(package_name: str) -> str:
    return f"""[package]
org = "test"
name = "test_project"
version = "0.1.0"
distribution = "2201.12.7"

[build-options]
observabilityIncluded = false
"""

def create_main_bal(main_content: str) -> str:
    return f"""{main_content}"""

def create_test_bal(test_content: str) -> str:
    return f"""{test_content}"""

def setup_build_ballerina(main_content: str, test_content: str) -> dict:
    """Set up temporary Ballerina project and run build with error handling"""
    try:
        # Create temporary directory with random UUID suffix
        package_name = f"test-project-{str(uuid4())[:8]}"

        with tempfile.TemporaryDirectory() as temp_dir:
            project_dir = Path(temp_dir) / package_name
            project_dir.mkdir()
            tests_dir = project_dir / "tests"
            tests_dir.mkdir()

            # Write project files
            (project_dir / "Ballerina.toml").write_text(create_ballerina_toml(package_name))
            (project_dir / "main.bal").write_text(create_main_bal(main_content))
            (tests_dir / "test.bal").write_text(create_test_bal(test_content))

            # Use BallerinaManager to get build status
            ballerina_manager = BallerinaManager(str(project_dir))
            build_result = ballerina_manager.get_build_status()

            return {
                "build_passed": build_result["success"],
                "build_stderr": build_result["stderr"],
                "compilation_errors": build_result["compilation_errors"],
                "package_name": package_name
            }
    except Exception as e:
        print(f"Error setting up Ballerina project: {e}")
        return {
            "build_passed": False,
            "build_stderr": f"Project setup error: {e}",
            "compilation_errors": [f"Project setup error: {e}"],
            "package_name": "unknown"
        }

def setup_build_test_ballerina(main_content: str, test_content: str) -> dict:
    """Set up temporary Ballerina project with tests and run build and test with error handling"""
    try:
        # Create temporary directory with random UUID suffix
        package_name = f"test-project-{str(uuid4())[:8]}"

        with tempfile.TemporaryDirectory() as temp_dir:
            project_dir = Path(temp_dir) / package_name
            project_dir.mkdir()
            tests_dir = project_dir / "tests"
            tests_dir.mkdir()

            # Write project files
            (project_dir / "Ballerina.toml").write_text(create_ballerina_toml(package_name))
            (project_dir / "main.bal").write_text(create_main_bal(main_content))
            (tests_dir / "test.bal").write_text(create_test_bal(test_content))

            # Use BallerinaManager to get build and test status
            ballerina_manager = BallerinaManager(str(project_dir))

            # Get build status first
            build_result = ballerina_manager.get_build_status()

            # Get test status only if build succeeds
            if build_result["success"]:
                test_result = ballerina_manager.get_test_status()
            else:
                test_result = {
                    "success": False,
                    "stdout": "",
                    "stderr": "Build failed, skipping tests",
                    "test_results": {"passed": 0, "failed": 0, "total": 0},
                    "compilation_errors": []
                }

            return {
                "build_passed": build_result["success"],
                "build_stderr": build_result["stderr"],
                "build_compilation_errors": build_result["compilation_errors"],
                "test_passed": test_result["success"],
                "test_stderr": test_result["stderr"],
                "test_results": test_result["test_results"],
                "test_compilation_errors": test_result["compilation_errors"],
                "package_name": package_name
            }
    except Exception as e:
        print(f"Error setting up Ballerina project with tests: {e}")
        return {
            "build_passed": False,
            "build_stderr": f"Project setup error: {e}",
            "build_compilation_errors": [f"Project setup error: {e}"],
            "test_passed": False,
            "test_stderr": f"Project setup error: {e}",
            "test_results": {"passed": 0, "failed": 0, "total": 0},
            "test_compilation_errors": [f"Project setup error: {e}"],
            "package_name": "unknown"
        }

print("✅ Ballerina project setup functions defined!")



# %%
# %%
# Define the code block delimiters
code_start = "<CODE>"
code_end = "</CODE>"
test_start = "<TESTS>"
test_end = "</TESTS>"

def extract_ballerina_code(response: str) -> str:
    """Extract Ballerina code from response - extracts content inside ```ballerina blocks within <CODE> tags"""
    # Extract everything between <CODE> and </CODE>
    pattern = rf"{re.escape(code_start)}(.*?){re.escape(code_end)}"
    match = re.search(pattern, response, re.DOTALL)
    if match:
        content = match.group(1).strip()
        # Now extract content from ```ballerina code block
        ballerina_pattern = r"```ballerina\s*(.*?)\s*```"
        ballerina_match = re.search(ballerina_pattern, content, re.DOTALL)
        if ballerina_match:
            return ballerina_match.group(1).strip()

    return ""

def extract_ballerina_tests(response: str) -> str:
    """Extract Ballerina tests from response - extracts content inside ```ballerina blocks within <TESTS> tags"""
    # Extract everything between <TESTS> and </TESTS>
    pattern = rf"{re.escape(test_start)}(.*?){re.escape(test_end)}"
    match = re.search(pattern, response, re.DOTALL)
    if match:
        content = match.group(1).strip()
        # Now extract content from ```ballerina code block
        ballerina_pattern = r"```ballerina\s*(.*?)\s*```"
        ballerina_match = re.search(ballerina_pattern, content, re.DOTALL)
        if ballerina_match:
            return ballerina_match.group(1).strip()

    return ""

def exact_ballerina_main_code(content: str) -> str:
    """Extract main Ballerina code content"""
    return extract_ballerina_code(content)

def exact_ballerina_test_code(content: str) -> str:
    """Extract test Ballerina code content"""
    return extract_ballerina_tests(content)



# %%
# %%
class BallerinaBaseEvaluator:
    def __init__(self, model_name="unsloth/Qwen2.5-Coder-7B-Instruct", max_seq_length=2048, debug_dir=None):
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.smoothing_function = SmoothingFunction().method1
        self.ballerina_manager = BallerinaManager()
        
        # Set up debug directory for persistence
        if debug_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            debug_dir = f"debug_outputs/{timestamp}"
        self.debug_dir = Path(debug_dir)
        self.debug_dir.mkdir(parents=True, exist_ok=True)

        # Load base model
        print(f"Loading base model: {model_name}")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=True,
        )

        # Set up chat template
        self.tokenizer = get_chat_template(
            self.tokenizer,
            chat_template="qwen-2.5",
        )

        # Enable inference mode
        FastLanguageModel.for_inference(self.model)
        print("Base model loaded successfully!")

    def _persist_evaluation_data(self, metric_type: str, problem_idx: int, prompt: str, reference: str, 
                               candidate_idx: int, raw_response: str, extracted_code: str, 
                               extracted_tests: str, result: Dict[str, Any], k_value: int):
        """Persist evaluation data for debugging purposes"""
        try:
            # Create directory structure
            metric_dir = self.debug_dir / f"{metric_type}_at_{k_value}"
            problem_dir = metric_dir / f"problem_{problem_idx+1:03d}"
            candidate_dir = problem_dir / f"candidate_{candidate_idx+1:03d}"
            candidate_dir.mkdir(parents=True, exist_ok=True)
            
            # Save prompt and reference (only once per problem)
            prompt_file = problem_dir / "prompt.txt"
            if not prompt_file.exists():
                prompt_file.write_text(prompt, encoding='utf-8')
                
            reference_file = problem_dir / "reference_answer.txt"
            if not reference_file.exists():
                reference_file.write_text(reference, encoding='utf-8')
            
            # Save candidate-specific data
            (candidate_dir / "raw_response.txt").write_text(raw_response, encoding='utf-8')
            
            if extracted_code:
                (candidate_dir / "extracted_code.bal").write_text(extracted_code, encoding='utf-8')
            
            if extracted_tests:
                (candidate_dir / "extracted_tests.bal").write_text(extracted_tests, encoding='utf-8')
            
            # Save results based on metric type
            if metric_type == "pass":
                result_file = candidate_dir / "test_result.json"
            else:
                result_file = candidate_dir / "compilation_result.json"
            
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2)
                
        except Exception as e:
            print(f"Warning: Failed to persist debug data: {e}")

    def _save_evaluation_metadata(self, dataset_size: int, num_samples: int, max_tokens: int):
        """Save evaluation metadata"""
        try:
            metadata = {
                "model_name": self.model_name,
                "evaluation_timestamp": datetime.now().isoformat(),
                "dataset_size": dataset_size,
                "num_samples_per_prompt": num_samples,
                "max_new_tokens": max_tokens,
                "system_prompt": SYSTEM_PROMPT
            }
            
            metadata_file = self.debug_dir / "metadata.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
                
            print(f"Debug data will be saved to: {self.debug_dir}")
        except Exception as e:
            print(f"Warning: Failed to save metadata: {e}")

    def preprocess_code(self, code: str) -> str:
        """Normalize code for comparison"""
        lines = [line.strip() for line in code.split('\n') if line.strip()]
        return '\n'.join(lines)

    def tokenize_for_bleu(self, text: str) -> List[str]:
        """Tokenize text for BLEU score calculation"""
        tokens = re.findall(r'\w+|[^\w\s]', text)
        return tokens

    def calculate_bleu_score(self, reference: str, candidate: str) -> float:
        """Calculate BLEU score between reference and candidate code"""
        ref_tokens = [self.tokenize_for_bleu(reference)]
        cand_tokens = self.tokenize_for_bleu(candidate)

        if not cand_tokens:
            return 0.0

        return sentence_bleu(ref_tokens, cand_tokens, smoothing_function=self.smoothing_function)

    def calculate_codebleu_score(self, reference: str, candidate: str) -> Dict[str, float]:
        """Calculate CodeBLEU score between reference and candidate code"""
        try:
            if not reference.strip() or not candidate.strip():
                return {
                    'codebleu': 0.0,
                    'ngram_match_score': 0.0,
                    'weighted_ngram_match_score': 0.0,
                    'syntax_match_score': 0.0,
                    'dataflow_match_score': 0.0
                }
            
            result = calc_codebleu(
                [reference],
                [candidate],
                lang="ballerina",
                weights=(0.25, 0.25, 0.25, 0.25)
            )
            
            return {
                'codebleu': result.get('codebleu', 0.0),
                'ngram_match_score': result.get('ngram_match_score', 0.0),
                'weighted_ngram_match_score': result.get('weighted_ngram_match_score', 0.0),
                'syntax_match_score': result.get('syntax_match_score', 0.0),
                'dataflow_match_score': result.get('dataflow_match_score', 0.0)
            }
        except Exception as e:
            print(f"Warning: CodeBLEU calculation failed: {e}")
            return {
                'codebleu': 0.0,
                'ngram_match_score': 0.0,
                'weighted_ngram_match_score': 0.0,
                'syntax_match_score': 0.0,
                'dataflow_match_score': 0.0
            }

    def calculate_rouge_scores(self, reference: str, candidate: str) -> Dict[str, float]:
        """Calculate ROUGE scores"""
        scores = self.rouge_scorer.score(reference, candidate)
        return {
            'rouge1': scores['rouge1'].fmeasure,
            'rouge2': scores['rouge2'].fmeasure,
            'rougeL': scores['rougeL'].fmeasure
        }

    def calculate_exact_match(self, reference: str, candidate: str) -> bool:
        """Calculate exact match after normalization"""
        ref_normalized = self.preprocess_code(reference)
        cand_normalized = self.preprocess_code(candidate)
        return ref_normalized == cand_normalized

    def calculate_edit_distance(self, reference: str, candidate: str) -> float:
        """Calculate normalized edit distance"""
        ref_normalized = self.preprocess_code(reference)
        cand_normalized = self.preprocess_code(candidate)

        if not ref_normalized and not cand_normalized:
            return 0.0

        max_len = max(len(ref_normalized), len(cand_normalized))
        if max_len == 0:
            return 0.0

        edit_dist = len(list(difflib.unified_diff(ref_normalized.split(), cand_normalized.split())))
        return 1.0 - (edit_dist / max_len)

    def check_ballerina_syntax(self, code: str) -> Dict[str, Any]:
        """Check if Ballerina code compiles and extract syntax information using BallerinaManager"""
        try:
            # Use the setup_build_ballerina function to create a proper project and test compilation
            build_result = setup_build_ballerina(code, "")

            compilation_success = build_result["build_passed"]
            error_message = build_result["build_stderr"]

            # If there are compilation errors, use them as the error message
            if build_result["compilation_errors"]:
                error_message = "; ".join(build_result["compilation_errors"])

        except Exception as e:
            # Fallback in case of any unexpected errors
            compilation_success = False
            error_message = f"Error during compilation check: {str(e)}"

        return {
            'compiles': compilation_success,
            'error_message': error_message,
            'has_function': 'function' in code.lower(),
            'has_service': 'service' in code.lower(),
            'has_import': 'import' in code.lower(),
            'has_return': 'return' in code.lower()
        }

    def calculate_compile_at_k(self, test_cases: List[str], candidates: List[List[str]], prompts: List[str], k: int = 1) -> float:
        """Calculate compile@k metric for compilation success only"""
        if len(test_cases) != len(candidates):
            raise ValueError("Number of test cases must match number of candidate lists")

        total_problems = len(test_cases)
        passed_problems = 0

        for i, (test_case, cands, prompt) in enumerate(zip(test_cases, candidates, prompts)):
            # Take first k candidates
            k_candidates = cands[:k]

            # Check if any of the k candidates compile successfully
            any_compiles = False
            for j, cand in enumerate(k_candidates):
                # Extract Ballerina code from LLM response
                extracted_code = exact_ballerina_main_code(cand)
                if extracted_code:  # Only check if we extracted valid code
                    syntax_check = self.check_ballerina_syntax(extracted_code)
                    
                    # Persist debug data
                    self._persist_evaluation_data(
                        metric_type="compile",
                        problem_idx=i,
                        prompt=prompt,
                        reference=test_case,  # Use test case as reference for debug
                        candidate_idx=j,
                        raw_response=cand,
                        extracted_code=extracted_code,
                        extracted_tests="",
                        result=syntax_check,
                        k_value=k
                    )
                    
                    if syntax_check['compiles']:
                        any_compiles = True
                        print(f"  Problem {i+1}: Candidate {j+1} compiled successfully!")
                        break
                    else:
                        print(f"  Problem {i+1}: Candidate {j+1} failed to compile")
                else:
                    # Persist data for failed extractions too
                    no_code_result = {
                        'compiles': False,
                        'error_message': 'No valid Ballerina code found in response',
                        'has_function': False,
                        'has_service': False,
                        'has_import': False,
                        'has_return': False
                    }
                    self._persist_evaluation_data(
                        metric_type="compile",
                        problem_idx=i,
                        prompt=prompt,
                        reference=test_case,  # Use test case as reference for debug
                        candidate_idx=j,
                        raw_response=cand,
                        extracted_code="",
                        extracted_tests="",
                        result=no_code_result,
                        k_value=k
                    )
                    
            if any_compiles:
                passed_problems += 1

        print(f"Compile@{k}: {passed_problems}/{total_problems} problems compiled")
        return passed_problems / total_problems if total_problems > 0 else 0.0

    def calculate_pass_at_k(self, test_cases: List[str], candidates: List[List[str]], prompts: List[str], k: int = 1) -> float:
        """Calculate pass@k metric for unit test success"""
        if len(test_cases) != len(candidates):
            raise ValueError("Number of test cases must match number of candidate lists")

        total_problems = len(test_cases)
        passed_problems = 0
        skipped_problems = 0

        for i, (test_case, cands, prompt) in enumerate(zip(test_cases, candidates, prompts)):
            # Take first k candidates
            k_candidates = cands[:k]
            
            # Check if any of the k candidates pass the unit tests
            any_passes = False
            for j, cand in enumerate(k_candidates):
                # Extract generated code from LLM response
                extracted_code = exact_ballerina_main_code(cand)
                
                if extracted_code:  # Only check if we extracted valid code
                    # Test the generated code against the provided test cases
                    test_result = setup_build_test_ballerina(extracted_code, test_case)
                    
                    # Persist debug data
                    self._persist_evaluation_data(
                        metric_type="pass",
                        problem_idx=i,
                        prompt=prompt,
                        reference=test_case,  # Use test case as reference for debug
                        candidate_idx=j,
                        raw_response=cand,
                        extracted_code=extracted_code,
                        extracted_tests=test_case,
                        result=test_result,
                        k_value=k
                    )
                    
                    if test_result["build_passed"] and test_result["test_passed"] and test_result["test_results"]["total"] > 0:
                        # All tests must pass
                        if test_result["test_results"]["failed"] == 0 and test_result["test_results"]["passed"] > 0:
                            any_passes = True
                            print(f"  Problem {i+1}: Candidate {j+1} passed all {test_result['test_results']['passed']} tests!")
                            break
                    else:
                        if not test_result["build_passed"]:
                            print(f"  Problem {i+1}: Candidate {j+1} failed to compile")
                        elif not test_result["test_passed"]:
                            print(f"  Problem {i+1}: Candidate {j+1} compiled but tests failed ({test_result['test_results']['failed']}/{test_result['test_results']['total']} failed)")
                else:
                    # Persist data for failed extractions too
                    no_code_result = {
                        'build_passed': False,
                        'test_passed': False,
                        'build_stderr': 'No valid Ballerina code found in response',
                        'test_stderr': 'No valid Ballerina code found in response',
                        'test_results': {'passed': 0, 'failed': 0, 'total': 0}
                    }
                    self._persist_evaluation_data(
                        metric_type="pass",
                        problem_idx=i,
                        prompt=prompt,
                        reference=test_case,  # Use test case as reference for debug
                        candidate_idx=j,
                        raw_response=cand,
                        extracted_code="",
                        extracted_tests=test_case,
                        result=no_code_result,
                        k_value=k
                    )

            if any_passes:
                passed_problems += 1

        effective_total = total_problems - skipped_problems
        print(f"Pass@{k}: {passed_problems}/{effective_total} problems passed ({skipped_problems} skipped)")
        return passed_problems / effective_total if effective_total > 0 else 0.0

    def generate_predictions(self, prompts: List[str], max_new_tokens: int = 1024, num_samples: int = 1) -> List[List[str]]:
        """Generate predictions for evaluation"""
        predictions = []

        for i, prompt in enumerate(prompts):
            print(f"Generating predictions for prompt {i+1}/{len(prompts)}")
            samples = []
            for _ in range(num_samples):
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ]

                inputs = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt"
                ).to("cuda")

                with torch.no_grad():
                    outputs = self.model.generate(
                        input_ids=inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=1.0,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id
                    )

                generated = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
                samples.append(generated.strip())

            predictions.append(samples)

        return predictions

    def load_evaluation_dataset(self, gist_url=None, max_samples=100):
        """Load the evaluation dataset"""
        if gist_url is None:
            gist_url = "https://gist.githubusercontent.com/xlight05/82fd685349270f67be89c2914c5d5b7a/raw/39d52ed1b310fb893865636a64af1cbf741724f7/tdd_test.json"

        print(f"Loading dataset from: {gist_url}")
        response = requests.get(gist_url)
        json_data = response.text
        data = json.loads(json_data)

        # Create dataset and limit to max_samples
        dataset = Dataset.from_list(data)
        if len(dataset) > max_samples:
            dataset = dataset.select(range(max_samples))

        print(f"Dataset loaded with {len(dataset)} samples")
        return dataset

    def evaluate_dataset(self, eval_dataset, num_samples: int = 1, max_new_tokens: int = 1024) -> Dict[str, Any]:
        """Comprehensive evaluation on a dataset
        
        Metrics calculated:
        - pass@k: Percentage of problems where at least one of k candidates passes unit tests
        - compile@k: Percentage of problems where at least one of k candidates compiles successfully
        - BLEU, ROUGE, CodeBLEU, exact match: Code generation quality metrics
        """

        # Extract prompts and test cases from dataset
        prompts = []
        test_cases = []

        for item in eval_dataset:
            prompts.append(item['prompt'])
            test_cases.append(item['test'])

        print(f"Evaluating base model on {len(prompts)} examples...")
        
        # Save evaluation metadata
        self._save_evaluation_metadata(len(prompts), num_samples, max_new_tokens)

        # Generate predictions
        predictions = self.generate_predictions(prompts, max_new_tokens, num_samples)

        # Calculate metrics
        results = {
            'model_name': self.model_name,
            'evaluation_type': 'base_model',
            'num_samples': len(prompts),
            'compilation_results': [],
            'pass_at_1': 0.0,
            'pass_at_5': 0.0 if num_samples >= 5 else None,
            'compile_at_1': 0.0,
            'compile_at_5': 0.0 if num_samples >= 5 else None,
            'syntax_features': defaultdict(list)
        }

        for i, (test_case, preds) in enumerate(zip(test_cases, predictions)):
            print(f"Evaluating sample {i+1}/{len(test_cases)}")

            # Use the first prediction for most metrics, extract code from LLM response
            pred = preds[0] if preds else ""
            extracted_pred_code = exact_ballerina_main_code(pred) if pred else ""

            # For this new approach, we don't have reference code to compare against
            # Instead, we focus on compilation and test success metrics
            # Skip BLEU, ROUGE, CodeBLEU, exact match, edit distance for now
            # since we don't have reference implementations to compare against

            # Syntax checking for all predictions
            compilation_results = []
            for pred in preds:
                # Extract Ballerina code from LLM response
                extracted_code = exact_ballerina_main_code(pred)
                if extracted_code:  # Only check if we extracted valid code
                    syntax_result = self.check_ballerina_syntax(extracted_code)
                else:
                    # No valid code found in response
                    syntax_result = {
                        'compiles': False,
                        'error_message': 'No valid Ballerina code found in response',
                        'has_function': False,
                        'has_service': False,
                        'has_import': False,
                        'has_return': False
                    }
                compilation_results.append(syntax_result)

                # Track syntax features
                for feature, value in syntax_result.items():
                    if feature != 'error_message':
                        results['syntax_features'][feature].append(value)

            results['compilation_results'].append(compilation_results)

        # Calculate compile@k metrics (compilation success only)
        print(f"\nCalculating Compile@k metrics (compilation success only)...")
        results['compile_at_1'] = self.calculate_compile_at_k(test_cases, predictions, prompts, k=1)
        if num_samples >= 5:
            results['compile_at_5'] = self.calculate_compile_at_k(test_cases, predictions, prompts, k=5)

        # Calculate pass@k metrics (unit test success)
        print(f"\nCalculating Pass@k metrics (unit test success)...")
        results['pass_at_1'] = self.calculate_pass_at_k(test_cases, predictions, prompts, k=1)
        if num_samples >= 5:
            results['pass_at_5'] = self.calculate_pass_at_k(test_cases, predictions, prompts, k=5)
        

        # Calculate aggregate statistics for compilation
        results['compilation_success_rate'] = np.mean([cr[0]['compiles'] for cr in results['compilation_results']])

        # Syntax feature statistics
        for feature, values in results['syntax_features'].items():
            if isinstance(values[0], bool):
                results[f'avg_{feature}'] = np.mean(values)

        return results

    def print_evaluation_summary(self, results: Dict[str, Any]):
        """Print a summary of evaluation results"""
        print("\n" + "="*60)
        print("BASE MODEL EVALUATION RESULTS")
        print("="*60)
        print(f"Model: {results['model_name']}")
        print(f"Evaluation Type: {results['evaluation_type']}")
        print(f"Number of Samples: {results['num_samples']}")

        print(f"\nCode Compilation & Testing:")
        print(f"  Pass@1 (Unit Tests):  {results['pass_at_1']:.4f}")
        if results['pass_at_5'] is not None:
            print(f"  Pass@5 (Unit Tests):  {results['pass_at_5']:.4f}")
        print(f"  Compile@1:            {results['compile_at_1']:.4f}")
        if results['compile_at_5'] is not None:
            print(f"  Compile@5:            {results['compile_at_5']:.4f}")
        print(f"  Compilation Success:  {results['compilation_success_rate']:.4f}")

        print(f"\nSyntax Features:")
        for feature in ['has_function', 'has_service', 'has_import', 'has_return']:
            if f'avg_{feature}' in results:
                print(f"  {feature.replace('_', ' ').title():15s}: {results[f'avg_{feature}']:.4f}")

        print("="*60)

    def run_sample_evaluation(self):
        """Run evaluation on sample prompts"""
        print("\n" + "="*60)
        print("SAMPLE PREDICTIONS FROM BASE MODEL")
        print("="*60)

        sample_prompts = [
            "Write a Ballerina function that calculates the sum of two numbers",
            "Create a Ballerina HTTP service that responds with 'Hello World'",
            "Write a Ballerina function that reads a file and returns its content",
            "Create a Ballerina function to connect to a database and fetch user data",
            "Write a Ballerina service that handles JSON requests and responses"
        ]

        sample_predictions = self.generate_predictions(sample_prompts, max_new_tokens=300, num_samples=1)

        for i, (prompt, preds) in enumerate(zip(sample_prompts, sample_predictions)):
            print(f"\n📝 Prompt {i+1}: {prompt}")
            print(f"🤖 Generated Response:\n{preds[0]}")

            # Extract and display the code
            extracted_code = exact_ballerina_main_code(preds[0])
            if extracted_code:
                print(f"🔍 Extracted Code:\n{extracted_code}")
                # Check syntax on extracted code
                syntax_result = self.check_ballerina_syntax(extracted_code)
                print(f"✅ Compiles: {syntax_result['compiles']}")
                if not syntax_result['compiles'] and syntax_result['error_message']:
                    print(f"❌ Error: {syntax_result['error_message']}")
            else:
                print("🔍 No valid Ballerina code found in response")
                print("❌ Error: Could not extract code from response")
            print("-" * 50)



# %%
# %%
import subprocess
import os

import urllib.request

# Download Ballerina .deb package
ballerina_url = "https://dist.ballerina.io/downloads/2201.12.7/ballerina-2201.12.7-swan-lake-linux-x64.deb"
deb_filename = "ballerina-2201.12.7-swan-lake-linux-x64.deb"

print("Downloading Ballerina...")
urllib.request.urlretrieve(ballerina_url, deb_filename)
print(f"✅ Downloaded {deb_filename}")

# Install the .deb package
print("Installing Ballerina...")
try:
    subprocess.run(["sudo", "dpkg", "-i", deb_filename], check=True)
    print("✅ Ballerina installed successfully!")
except subprocess.CalledProcessError as e:
    print(f"Installation failed: {e}")
    print("Trying to fix dependencies...")
    subprocess.run(["sudo", "apt-get", "-f", "install"], check=True)

# Test Ballerina version
print("Testing Ballerina installation...")
try:
    result = subprocess.run(["bal", "-v"], capture_output=True, text=True, check=True)
    print("✅ Ballerina version:")
    print(result.stdout)
except subprocess.CalledProcessError as e:
    print(f"❌ Failed to run 'bal -v': {e}")
except FileNotFoundError:
    print("❌ 'bal' command not found. Installation may have failed.")

# Clean up downloaded file
os.remove(deb_filename)
print(f"🧹 Cleaned up {deb_filename}")



# %%
# %%
modelName = "xlight05/base_test_4_grpo_16bit_vllm"
def main():
    """Main evaluation function"""
    print("Starting Ballerina Base Model Evaluation")
    print("="*60)

    # Initialize evaluator
    evaluator = BallerinaBaseEvaluator(modelName)

    # Load evaluation dataset (limit to 50 samples for faster evaluation)
    eval_dataset = evaluator.load_evaluation_dataset(max_samples=100)

    # Run comprehensive evaluation
    print("\nRunning comprehensive evaluation...")
    evaluation_results = evaluator.evaluate_dataset(
        eval_dataset,
        num_samples=5,  # Generate 5 samples per prompt for pass@5 metric
        max_new_tokens=1024
    )

    # Print results
    evaluator.print_evaluation_summary(evaluation_results)

    # Save results
    output_file = 'base_model_evaluation_results.json'
    with open(output_file, 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        results_for_json = {}
        for key, value in evaluation_results.items():
            if isinstance(value, np.ndarray):
                results_for_json[key] = value.tolist()
            elif isinstance(value, (np.float64, np.float32)):
                results_for_json[key] = float(value)
            elif isinstance(value, (np.int64, np.int32)):
                results_for_json[key] = int(value)
            elif isinstance(value, defaultdict):
                results_for_json[key] = dict(value)
            else:
                results_for_json[key] = value

        json.dump(results_for_json, f, indent=2)

    print(f"\n📊 Evaluation results saved to '{output_file}'")

    # # Run sample evaluation
    # evaluator.run_sample_evaluation()

    # print("\n🎉 Base model evaluation completed!")


if __name__ == "__main__":
    main()



# %%



