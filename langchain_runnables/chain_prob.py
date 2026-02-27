import random


class NakliLLM:
    def __init__(self):
        print("LLM created")

    def predict(self, prompt):
        response_list = [
            "Islamabad is the capital of Pakistan",
            "PSL is a cricket league",
            "AI stands for Artificial Intelligence",
        ]

        return {"response": random.choice(response_list)}


llm = NakliLLM()
# result = llm.predict('What is the capital of Pakistan')
# print(result)


class NakliPromptTemplate:
    def __init__(self, template, input_variables):
        self.template = template
        self.input_variables = input_variables

    def format(self, input_dict):
        return self.template.format(**input_dict)


template = NakliPromptTemplate(
    template="Write a {length} poeam about {topic}", input_variables=["length", "topic"]
)

# prompt = template.format({"length": "short", "topic": "pakistan"})
# result = llm.predict(prompt)
# print(result)


class NakliLLMChain:
    def __init__(self, llm, prompt):
        self.llm = llm
        self.prompt = prompt

    def run(self, input_dict):
        final_prompt = self.prompt.format(input_dict)
        result = self.llm.predict(final_prompt)

        return result["response"]


chain = NakliLLMChain(llm, template)
result = chain.run({"length": "short", "topic": "india"})
print(result)
