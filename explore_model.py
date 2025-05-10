from transformers import AutoTokenizer
from model import Transformer, TransformerModelArgs

def main():
    tokenizer_name = "unsloth/Mistral-Nemo-Base-2407-bnb-4bit"
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        vocab_size = tokenizer.vocab_size
        print(f"Successfully loaded tokenizer: {tokenizer_name}")
        print(f"Vocabulary size: {vocab_size}")
    except Exception as e:
        print(f"Error loading tokenizer '{tokenizer_name}': {e}")
        print("Please ensure the tokenizer is accessible (e.g., internet connection on node or pre-downloaded).")
        print("Using a default vocab_size = 32000 for now. This might affect the parameter count if incorrect.")
        vocab_size = 32000

    model_config = TransformerModelArgs(
        dim=4096,
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        multiple_of=1024,
        rope_theta=500000.0,
        vocab_size=vocab_size,
        seq_len=4096
    )
    print("\nModel Configuration:")
    for arg, value in vars(model_config).items():
        print(f"  {arg}: {value}")

    print("\nInstantiating the model...")
    try:
        model = Transformer(model_config)
        print("Model instantiated successfully.")
    except Exception as e:
        print(f"Error instantiating model: {e}")
        print("Ensure 'model.py' is in the correct directory and there are no issues with its code.")
        return

    print("\nCalculating model parameters...")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nTotal Trainable Model Parameters: {total_params:,}")


if __name__ == "__main__":
    main()
