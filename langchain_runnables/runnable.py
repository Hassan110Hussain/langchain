import random
from abc import ABC, abstractmethod
from unittest import result


class Runnable(ABC):
    @abstractmethod
    def invoke(input_data):
        pass


class NakliLLM(Runnable):
    def __init__(self):
        print("LLM created")

    def invoke(self, prompt):
        response_list = [
            # "Islamabad is the capital of Pakistan",
            # "PSL is a cricket league",
            # "AI stands for Artificial Intelligence",
            "Why don’t football players get hot? Because they have lots of fans!",
            "Why did the football coach go to the bank? To get his quarter back!",
            "Why was the football stadium so hot? Because all the fans left!"
        ]

        return {"response": random.choice(response_list)}

    # def predict(self, prompt):
    #     response_list = [
    #         "Islamabad is the capital of Pakistan",
    #         "PSL is a cricket league",
    #         "AI stands for Artificial Intelligence",
    #     ]

    #     return {"response": random.choice(response_list)}  predict func depreciated


class NakliPromptTemplate(Runnable):
    def __init__(self, template, input_variables):
        self.template = template
        self.input_variables = input_variables

    def invoke(self, input_dict):
        return self.template.format(**input_dict)

    # def format(self, input_dict):
    #     return self.template.format(**input_dict)  format func depreciated


class nakliStrOutputParser(Runnable):
    def __init__(self):
        pass

    def invoke(self, input_data):
        return input_data["response"]


class RunnableConnector(Runnable):
    def __init__(self, runnable_list):
        self.runnable_list = runnable_list

    def invoke(self, input_data):
        result = input_data
        for runnable in self.runnable_list:
            result = runnable.invoke(result)

        return result


template = NakliPromptTemplate(
    template="Write a {length} poeam about {topic}", input_variables=["length", "topic"]
)

# parser = nakliStrOutputParser()
# llm = NakliLLM()
# chain = RunnableConnector([template, llm, parser])
# result = chain.invoke({"length": "long", "topic": "pakistan"})
# print(result)

template1 = NakliPromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)

template2 = NakliPromptTemplate(
    template='Explain the following joke {response}',
    input_variables=['response']
)

parser = nakliStrOutputParser()
llm = NakliLLM()
chain1 = RunnableConnector([template1, llm])
chain2 = RunnableConnector([template2, llm, parser])

final_chain = RunnableConnector([chain1, chain2])
result = final_chain.invoke({'topic': 'football'})
print(result)
