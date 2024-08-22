class Tokenizer:
    def __init__(self, text: str = ""):
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.stoi = {char: i for i, char in enumerate(chars)}
        self.itos = {i: char for i, char in enumerate(chars)}

        # Add UNK token
        self.stoi["<UNK>"] = self.vocab_size
        self.itos[self.vocab_size] = "<UNK>"
    
    def encode(self, text: str) -> list[int]:
        return [self.stoi.get(char, self.stoi["<UNK>"]) for char in text]
    
    def decode(self, tokens: list[int]) -> str:
        return "".join([self.itos.get(i, "<UNK>") for i in tokens])
        