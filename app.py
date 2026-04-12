from datetime import datetime
from typing import Literal
from agents import function_tool

from openai import OpenAI, AsyncOpenAI
from agents import Agent, Runner, OpenAIChatCompletionsModel
from agents import set_tracing_disabled
from agents import Agent, Runner, ModelSettings
from agents import RunResult   # ✅ FIXED
from pydantic import BaseModel, Field
import asyncio   # ✅ FIXED

set_tracing_disabled(True)

def get_ollama_model():
    client = AsyncOpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama"
    )

    local_model = OpenAIChatCompletionsModel(
        model="minimax-m2.5:cloud",
        openai_client=client
    )

    return local_model


MODEL = get_ollama_model()

class TicketClassification(BaseModel):
    category: Literal["billing", "technical", "account", "general"] = Field()
    priority: Literal["P1-critical", "P2-high", "P3-medium", "P4-low"] = Field()
    sentiment: Literal["angry", "frustrated", "neutral", "positive"] = Field()
    summary: str = Field()


@function_tool
def lookup_customer(email: str) -> str:
    customers = {
        "ahmed@example.com": {
            "name": "Ahmed Hassan",
            "plan": "Enterprise",
            "since": "2023-01",
            "mrr": "$299/mo",
            "tickets_open": 2,
        },
        "sara@startup.io": {
            "name": "Sara Khan",
            "plan": "Pro",
            "since": "2024-06",
            "mrr": "$49/mo",
            "tickets_open": 0,
        },
    }
    customer = customers.get(email.lower())
    if not customer:
        return f"No customer found with email: {email}"
    return (
        f"Customer: {customer['name']}\n"
        f"Plan: {customer['plan']} ({customer['mrr']})\n"
        f"Customer since: {customer['since']}\n"
        f"Open tickets: {customer['tickets_open']}"
    )


@function_tool
def check_service_status(service: str) -> str:
    statuses = {
        "api": "Operational (99.98% uptime, 45ms avg latency)",
        "dashboard": "Degraded (slow loading, team investigating)",
        "billing": "Operational",
        "auth": "Operational",
    }
    status = statuses.get(service.lower())
    if not status:
        return f"Unknown service: {service}. Available: {list(statuses.keys())}"
    return f"{service}: {status}"


@function_tool
def create_ticket(customer_email: str, category: str, priority: str, description: str) -> str:
    ticket_id = f"TKT-{abs(hash(description)) % 100000:05d}"
    return (
        f"Ticket created!\n"
        f"ID: {ticket_id}\n"
        f"Customer: {customer_email}\n"
        f"Category: {category} | Priority: {priority}\n"
        f"Description: {description[:100]}...\n"
        f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    )


@function_tool
def search_knowledge_base(query: str) -> str:
    articles = {
        "password": "Reset password instructions...",
        "billing": "Billing info...",
        "api": "API docs...",
        "export": "Export guide...",
    }
    for key, article in articles.items():
        if key in query.lower():
            return article
    return "No relevant articles found. Escalate to human agent."


classifier_agent = Agent(
    name="Ticket Classifier",
    instructions="""Return valid JSON only.""",
    model=MODEL,
    output_type=TicketClassification,
    model_settings=ModelSettings(temperature=0.1),
)


def support_instructions(context, agent):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
    return f"You are support agent. Time: {current_time}"


support_agent = Agent(
    name="cloudsync_support_agent",
    instructions=support_instructions,
    model=MODEL,
    model_settings=ModelSettings(temperature=0.3, max_tokens=1000),
    tools=[
        lookup_customer,
        check_service_status,
        create_ticket,
        search_knowledge_base,
        classifier_agent.as_tool(
            tool_name="classify_ticket",
            tool_description="Classify a customer message",
        ),
    ],
)


async def handle_customer(message: str) -> None:
    print(f"\n{'='*70}")
    print(f"Customer: {message}")
    print(f"{'='*70}")

    result: RunResult = await Runner.run(support_agent, message)

    print(f"\nAgent Response:\n{result.final_output}")
    print(f"\nAgent: {result.last_agent.name}")
    print(f"Items generated: {len(result.new_items)}")


# ✅ FIXED: main OUTSIDE function
async def main():
    await handle_customer(
        "Hi, I'm ahmed@example.com. The dashboard has been super slow today. "
        "Is something wrong with the system?"
    )


# ✅ FIXED
if __name__ == "__main__":
    asyncio.run(main())