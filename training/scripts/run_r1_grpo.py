import logging
import os
from dataclasses import dataclass
from datetime import datetime
import logging
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
import random
import re 
import torch
from transformers.trainer_utils import get_last_checkpoint
from transformers import AutoTokenizer
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer, get_peft_config, ModelConfig, TrlParser


########################
# Custom dataclasses
########################
@dataclass
class ScriptArguments:
    dataset_id_or_path: str = "Jiayi-Pan/Countdown-Tasks-3to4"
    dataset_splits: str = "train"
    tokenizer_name_or_path: str = None


########################
# Setup logging
########################
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger.addHandler(handler)

########################
# Helper functions
########################

def format_reward_func(completions, target, **kwargs):
    """
    Format: <think>...</think><answer>...</answer>
    Args:
        completions (list[str]): Generated outputs
        target (list[str]): Expected answers
      
      Returns:
          list[float]: Reward scores
    """
    rewards = []

    for completion, gt in zip(completions, target):

      try:
        # add synthetic <think> as its already part of the prompt and prefilled for the assistant to more easily match the regex
        completion = "<begin_of_thought>" + completion
        if random.random() < 0.1:  # 1% chance to write samples into a file
          os.makedirs("completion_samples", exist_ok=True)
          log_file = os.path.join("completion_samples", "completion_samples.txt")
          with open(log_file, "a") as f:
            f.write(f"\n\n==============\n")
            f.write(completion)
        
        # Check if the format is correct
        #regex = r"(?s)^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>(?=.*\\boxed\{((?:[^{}]|\{[^}]*\})*)\})"
        regex = r"(?s)^<begin_of_thought>((?!<begin_of_thought>).*?)<end_of_thought>.*?<begin_of_solution>((?!<begin_of_solution>).*?)<end_of_solution>$"


        match = re.search(regex, completion, re.DOTALL) 
        # if the format is not correct, reward is 0
        if match is None or len(match.groups()) != 2:
            rewards.append(0.0)
        else:
            rewards.append(1.0)
      except Exception:
        rewards.append(0.0)
    return rewards

def convert_latex_to_python(latex_str):
    """
    Convert common LaTeX math expressions to Python-valid syntax.
    """
    # Replace LaTeX fraction commands with Python's fraction syntax
    latex_str = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'(\1 / \2)', latex_str)
    
    # Replace LaTeX left and right brackets with regular parentheses
    latex_str = latex_str.replace(r'\left(', '(').replace(r'\right)', ')')
    
    # Handle other possible LaTeX commands or formatting if necessary (extend as needed)
    # For example, you can add more replacements here for other LaTeX functions.
    
    return latex_str

def verify_equation(equation):
    """
    Verifies if the equation is true after parsing LaTeX-like math expressions.
    
    The equation is expected to be in the form: 'LHS = RHS', where LHS and RHS are mathematical expressions.
    """
    # Split the equation into left-hand side and right-hand side parts
    lhs, rhs = equation.split('=')
    
    # Clean up the LaTeX expressions
    lhs = convert_latex_to_python(lhs)
    rhs = convert_latex_to_python(rhs)
    
    try:
        # Evaluate both sides and compare
        return eval(lhs) == eval(rhs)
    except Exception as e:
        # In case of invalid syntax or error during evaluation
        print(f"Error evaluating the equation: {e}")
        return False

def equation_reward_func(completions, target, nums, **kwargs):
    """
    Evaluates completions based on:
    2. Mathematical correctness of the answer

    Args:
        completions (list[str]): Generated outputs
        target (list[str]): Expected answers
        nums (list[str]): Available numbers
    
    Returns:
        list[float]: Reward scores
    """
    rewards = []
    for completion, gt, numbers in zip(completions, target, nums):
      try:
        # add synthetic <think> as its already part of the prompt and prefilled for the assistant to more easily match the regex
        completion = "<begin_of_thought>" + completion
        #logger.info(f"FULL COMPLETION: {completion}")
        # Check if the format is correct
        match = re.search(r"(?s)^<begin_of_thought>((?!<begin_of_thought>).*?)<end_of_thought>.*?<begin_of_solution>((?!<begin_of_solution>).*?)<end_of_solution>$", completion)
        #regex = r"(?s)^<begin_of_thought>((?!<begin_of_thought>).*?)<end_of_thought>.*?<begin_of_solution>((?!<begin_of_solution>).*?)<end_of_solution>$"
        if match is None:
            rewards.append(0.0)
            continue
        # Extract the "answer" part from the completion
        equation = match.group(2).strip()
        logger.info(f"EQUATION: {equation}")
        # Extract all numbers from the equation
        pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
        matches = re.findall(pattern, equation)
        equation = matches[0]
        """
        if "=" in equation:
            sides = equation.split("=")
            if sides[1].strip() == gt:
                equation = sides[0]
            else:
                rewards.append(0.0)
                continue
        """

        used_numbers = [int(n) for n in re.findall(r'\d+', equation)]
        logger.info(f"Numbers: {used_numbers}")
        # Check if all numbers are used exactly once
        if sorted(used_numbers) != sorted(numbers.append(gt)):
            rewards.append(0.0)
            continue
        logger.info(f"NUMBER CHECK PASSED")

        if verify_equation(equation):
            rewards.append(1.0)
            logger.info("WOOHOO")
        else:
            rewards.append(0.0)
            logger.info("RATS")
        continue

        # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
        allowed_pattern = r'^[\d+\-*/().\s]+$'
        if not re.match(allowed_pattern, equation):
           rewards.append(0.0)
           continue
        logger.info(f"PATTERN CHECK PASSED")
        # Evaluate the equation with restricted globals and locals
        result = eval(equation, {"__builtins__": None}, {})
        logger.info(f"RESULT: {result} VS {gt}")
        # Check if the equation is correct and matches the ground truth
        if abs(float(result) - float(gt)) < 1e-5:
            rewards.append(1.0)
            logger.info(f"WE DID IT")
            if random.random() < 0.10:  # 10% chance to write fully successful samples into a file
                os.makedirs("completion_samples", exist_ok=True)
                log_file = os.path.join("completion_samples", "success_completion_samples.txt")
                with open(log_file, "a") as f:
                    f.write(f"\n\n==============\n")
                    f.write(completion)
        else:
            rewards.append(0.1)
      except Exception:
            # If evaluation fails, reward is 0
            rewards.append(0.0) 
    return rewards

def get_checkpoint(training_args: GRPOConfig):
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    return last_checkpoint


def grpo_function(
    model_args: ModelConfig, script_args: ScriptArguments, training_args: GRPOConfig
):
    #########################
    # Log parameters
    #########################
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    ################
    # Load tokenizer
    ################
    tokenizer = AutoTokenizer.from_pretrained(
        (
            script_args.tokenizer_name_or_path
            if script_args.tokenizer_name_or_path
            else model_args.model_name_or_path
        ),
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ###############
    # Load datasets
    ###############
    # Load dataset from Hugging Face Hub
    dataset = load_dataset(script_args.dataset_id_or_path, split=script_args.dataset_splits)
    # select a random subset of 50k samples
    dataset = dataset.shuffle(seed=42).select(range(50000))

    #####################
    # Prepare and format dataset
    #####################

    # gemerate r1 prompt with a prefix for the model to already start with the thinking process
    def generate_r1_prompt(numbers, target):
        r1_prefix = [{
            "role": "system",
            "content": "I am a Red Hat® Instruct Model, an AI language model developed by Red Hat and IBM Research based on the granite-3.1-8b-base model. My primary role is to serve as a chat assistant."
          },
          { 
            "role": "user",
            #"content": f"Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <begin_of_thought> <end_of_thought> tags. And return the final equation and answer in \\boxed{{}}, for example  \\boxed{{95 - \left( \\frac{{21}}{{3}} \\right) = 88}}, within the <begin_of_solution>."
            "content": f"Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <begin_of_thought> <end_of_thought> tags. And return the final equation and answer in <begin_of_solution> <end_of_solution> tags within \\boxed{{}}, for example <begin_of_solution> \\boxed{{95 - \left( \\frac{{21}}{{3}} \\right) = 88}} <end_of_solution>. Think step by step inside <begin_of_thought> <end_of_thought> tags."
            #"content": ("Your role as an assistant involves thoroughly exploring questions through a systematic"
            #" long thinking process before providing the final precise and accurate solutions. This requires engaging"
            #" in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing,"
            #" and iteration to develop well-considered thinking process. Please structure your response into two main"
            #" sections: Thought and Solution. In the Thought section, detail your reasoning process using the specified"
            #" format: <|begin_of_thought|> {thought with steps separated with '\\n\\n'} <|end_of_thought|> Each step should"
            #" include detailed considerations such as analisying questions, summarizing relevant findings, brainstorming new"
            #" ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps. In the"
            #" Solution section, based on various attempts, explorations, and reflections from the Thought section,"
            #" present the final solution that you deem correct. The solution should remain a logical, accurate, concise expression"
            #" formatted as follows: <|begin_of_solution|>"
            #" {final formatted, precise, and clear solution} <|end_of_solution|> Now, try to solve the following question through"
            #" the above guidelines:") + f"Using the numbers {numbers}, create an expression that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <begin_of_thought> <end_of_thought> tags. And return the final expression (just the expression) in <begin_of_solution> <end_of_solution> tags, for example <begin_of_solution> (1 + 2) / 3 <end_of_solution>. Solutions will not be computed unless within solution tags."
          },
          {
            "role": "assistant",
            #"content": "Let me solve this step by step.\n<think>"
            "content": "Let me solve this step by step.\n<begin_of_thought>"
          }]
        return {"prompt": tokenizer.apply_chat_template(r1_prefix, tokenize=False, continue_final_message=True), "target": target, "nums": numbers}

    # convert our dataset to the r1 prompt
    dataset = dataset.map(lambda x: generate_r1_prompt(x["nums"], x["target"]))

    # split the dataset into train and test
    train_test_split = dataset.train_test_split(test_size=0.1)

    train_dataset = train_test_split["train"]
    test_dataset = train_test_split["test"]

    #########################
    # Instantiate DPO trainer
    #########################

    trainer = GRPOTrainer(
      model=model_args.model_name_or_path,
      reward_funcs=[format_reward_func, equation_reward_func],
      args=training_args,
      train_dataset=train_dataset,
      eval_dataset=test_dataset,
      peft_config=get_peft_config(model_args),
    )


    ###############
    # Training loop
    ###############
    # Check for last checkpoint
    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}.")

    # Train the model
    logger.info(
        f'*** Starting training {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} for {training_args.num_train_epochs} epochs***'
    )
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
    # Log and save metrics
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("*** Training complete ***")

    ##################################
    # Save model and create model card
    ##################################

    logger.info("*** Save model ***")
    trainer.model.config.use_cache = True
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")
    training_args.distributed_state.wait_for_everyone()  # wait for all processes to load

    tokenizer.save_pretrained(training_args.output_dir)
    logger.info(f"Tokenizer saved to {training_args.output_dir}")

    # Save everything else on main process
    if trainer.accelerator.is_main_process:
        trainer.create_model_card({"tags": ["rl","grpo", "tutorial", "philschmid"]})
    # push to hub if needed
    if training_args.push_to_hub is True:
        logger.info("Pushing to hub...")
        trainer.push_to_hub()

    logger.info("*** Training complete! ***")


def main():
    parser = TrlParser((ModelConfig, ScriptArguments, GRPOConfig))
    model_args, script_args, training_args = parser.parse_args_and_config()

    # Run the main training loop
    grpo_function(model_args, script_args, training_args)


if __name__ == "__main__":
    main()