import json
from agent import PainPointAgent

if __name__ == "__main__":
    # --- Example Usage ---

    # 1. Initialize the agent with the knowledge base (alpha=0.2 for keyword vs semantic balance)
    agent = PainPointAgent("knowledge_base.json", alpha=0.2)

    # 2. Define user's pain points
    user_pain_point_1 = {
        "pain_point_description": "Our support agents are overwhelmed by the high volume of repetitive questions."
    }

    user_pain_point_2 = {
        "pain_point_description": "Manually analyzing thousands of open-ended survey responses for common themes is too time-consuming."
    }

    user_pain_point_3 = {
        "pain_point_description": "Thật khó để có được cái nhìn toàn diện về lịch sử tương tác của khách hàng khi họ liên hệ với chúng tôi."
    }

    # 3. Get suggestions from the agent with detailed scoring
    print("--- Analyzing Pain Point 1 ---")
    solutions_1 = agent.find_solutions(user_pain_point_1, k=3)
    print(json.dumps(solutions_1, indent=2, ensure_ascii=False))

    print("\n--- Analyzing Pain Point 2 ---")
    solutions_2 = agent.find_solutions(user_pain_point_2, k=3)
    print(json.dumps(solutions_2, indent=2, ensure_ascii=False))

    print("\n--- Analyzing Pain Point 3 ---")
    solutions_3 = agent.find_solutions(user_pain_point_3, k=3)
    print(json.dumps(solutions_3, indent=2, ensure_ascii=False))
