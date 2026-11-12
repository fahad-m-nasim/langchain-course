# from _typeshed import OpenBinaryMode

from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model

from langchain.tools import tool

from langchain.messages import HumanMessage, SystemMessage, ToolMessage

from langsmith import traceable


MAX_ITERATIONS = 10
MODEL = "qwen3:1.7b"

@tool
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog."""
    print(f"        >>Executing get_product_price(product='{product}')")
    prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}
    return prices.get(product, 0)

@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price and return a final price.
        Available tiers: bronze, silver, gold
    """
    print(f"        >>Executing apply_discount(price='{price}', discount_tier='{discount_tier}')")
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount / 100 ), 2)



# ---- Agent Loop ---- 
@traceable(name="Langchain Agent Loop")
def run_agent(question: str):
    tools = [get_product_price, apply_discount]
    tools_dict = {t.name: t for t in tools}

    llm = init_chat_model(model=f"ollama:{MODEL}", temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    print(f"Question: {question}")
    print("=" * 60)

    messages = [
        SystemMessage(
            content=(
                "You are a helful shopping assistant. "
                "You have access to a product catalog tool "
                "and a discount tool.\n\n"
                "STRICT RULES - you must follow these rules exactly:\n"
                "1. NEVER guess or assume any product price. "
                "You must call get_product_price function first to get the real price of the product.\n"
                "2. Only call apply_discount AFTER you have received "
                "a price from get_product_price. Pass the exact price "
                "returned by get_product_price - do NOT pass a made-up number.\n"
                "3. NEVER calculate discounts yourself using math."
                "Always use apply_discount tool.\n"
                "4. If a user does not specify a discount tier, "
                "ask them, which tier to use - Do not assume one."
            )
        ),
        HumanMessage(
            content=question
        )
    ]

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {iteration} ---")

        ai_message = llm_with_tools.invoke(messages)
        tool_calls = ai_message.tool_calls

        # If no tool call, then this is the final answer.

        if not tool_calls:
            print(f"\nFinal Answer: {ai_message.content}")
            return ai_message.content

        # Process only first tool call - force one tool call per iteration
        tool_call = tool_calls[0]
        tool_name = tool_call["name"]
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")

        print(f"    [Tool Selected] {tool_name} with args: {tool_args}")

        tool_to_use = tools_dict.get(tool_name)

        if tool_to_use is None:
            raise ValueError(f"tool to use {tool_name} is found null")

        observation = tool_to_use.invoke(tool_args)

        print(f"    [Tool Result] {observation}")

        messages.append(ai_message)
        messages.append(
            ToolMessage(content=str(observation), tool_call_id=tool_call_id)
        )


    print("Error: Max iterations reached without a final answer")
    return None

if __name__ == "__main__":
    print(f"Hello LangChain Agent (.bind_tools)!")
    print()
    result = run_agent("What is a price of a laptop after applying gold discount!")

