from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
import random
from operator import itemgetter
from langchain.schema.runnable import RunnableLambda

load_dotenv()

model = ChatOpenAI(temperature=1.0, model="gpt-4")

# 創作料理の食材の提案チェーン
suggest_ingredients_prompt = ChatPromptTemplate.from_template(
    "創作料理の食材を{num_ingredients}個提案してください"
)
suggest_ingredients_function = {
    "name": "set_ingredients",
    "description": "set ingredients list",
    "parameters": {
        "type": "object",
        "properties": {"ingredients": {"type": "array", "items": {"type": "string"}}},
        "required": ["ingredients"],
    },
}
suggest_ingredients_chain = (
    suggest_ingredients_prompt
    | model.bind(
        function_call={"name": suggest_ingredients_function["name"]},
        functions=[suggest_ingredients_function],
    )
    | JsonOutputFunctionsParser()
)

# 創作料理の提案チェーン
suggest_dish_prompt = ChatPromptTemplate.from_template(
    "以下を食材として利用した創作料理を提案してください: {ingredients}"
)
suggest_dish_function = {
    "name": "set_dish",
    "description": "set dish",
    "parameters": {
        "type": "object",
        "properties": {
            "dish_name": {"type": "string"},
            "dish_description": {"type": "string"},
        },
        "required": ["name", "description"],
    },
}
suggest_dish_chain = (
    suggest_dish_prompt
    | model.bind(
        function_call={"name": suggest_dish_function["name"]},
        functions=[suggest_dish_function],
    )
    | JsonOutputFunctionsParser()
)


# listのなかからランダムに指定個数の要素を取り出し新しいlistを作成する関数
def random_choice_list(lst, num):
    return random.sample(lst, num)


random_choise_chain = {
    "ingredients": itemgetter("ingredients")
    | RunnableLambda(lambda x: random_choice_list(x, 3))
}

chef_chain = random_choise_chain | suggest_dish_chain

# 食材の提案チェーンと創作料理の提案チェーンを合成
total_chain = suggest_ingredients_chain | {
    "chef1": chef_chain,
    "chef2": chef_chain,
    "chef3": chef_chain,
}

# チェーンを実行
result = total_chain.invoke({"num_ingredients": "5"})
print(result)
