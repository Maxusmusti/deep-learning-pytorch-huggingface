# Ensure conda is initialized
echo "Initializing Conda..."
eval "$(conda shell.bash hook)" || { echo "Failed to initialize Conda"; exit 1; }

# Set up the Conda environment
echo "Setting up Conda environment..."

# create the environment if it doesn't exist
if ! conda env list | grep -q "mpo"; then
    conda create -n mpo python=3.11 -y || { echo "Failed to create Conda environment"; exit 1; }
    conda activate mpo || { echo "Failed to activate Conda environment"; exit 1; }

    pip install wheel
    pip install flash-attn

    pip install  --upgrade \
      "transformers==4.48.1" \
      "datasets==3.1.0" \
      "accelerate==1.3.0" \
      "hf-transfer==0.1.9" \
      "deepspeed==0.15.4" \
      "trl==0.14.0"

    pip install "vllm==0.7.0"
    pip install latex2sympy2
    pip install word2number
    pip install timeout_decorator

    echo "Conda environment setup complete"
fi

conda activate mpo || { echo "Failed to activate Conda environment"; exit 1; }
