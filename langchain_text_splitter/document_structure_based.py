from langchain.text_splitter import Language, RecursiveCharacterTextSplitter

text = """class BankAccount:
    def __init__(self, owner, balance=0):
        self.owner = owner
        self.balance = balance

    def deposit(self, amount):
        if amount > 0:
            self.balance += amount
            print(f"Deposited {amount}. New balance is {self.balance}")
        else:
            print("Deposit amount must be positive.")

    def withdraw(self, amount):
        if amount > self.balance:
            print("Insufficient funds.")
        elif amount <= 0:
            print("Withdrawal amount must be positive.")
        else:
            self.balance -= amount
            print(f"Withdrew {amount}. New balance is {self.balance}")

    def __str__(self):
        return f"Account owner: {self.owner}, Balance: {self.balance}"


# Example usage
account = BankAccount("Alice", 100)
print(account)

account.deposit(50)
account.withdraw(30)
account.withdraw(200)
"""

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=100,
    chunk_overlap=0,
)

chunks = splitter.split_text(text)

print(chunks[0])
print(len(chunks))