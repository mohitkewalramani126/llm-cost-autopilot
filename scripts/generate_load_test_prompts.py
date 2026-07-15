"""Generates a large set of novel prompts for Phase 6 load testing.

Unlike data/labeled_prompts.csv (hand-labeled for classifier training),
these prompts are unlabeled -- the point of this load test is to see how
the already-trained classifier and router perform on traffic they've
never encountered, not to add more training data. Output has no overlap
with the training set.
"""

import csv
import random
from pathlib import Path

random.seed(42)

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "load_test_prompts.csv"

LANGUAGES = ["Python", "JavaScript", "Go", "Rust", "TypeScript", "Java", "C++", "SQL", "Ruby", "Swift"]
CONCEPTS_A = ["a list", "a stack", "synchronous code", "a class", "a mutex", "REST", "an array", "a thread"]
CONCEPTS_B = ["a set", "a queue", "asynchronous code", "a struct", "a semaphore", "GraphQL", "a linked list", "a process"]
TASKS = [
    "reverses a string", "checks if a number is prime", "merges two sorted lists",
    "finds the longest common subsequence", "flattens a nested dictionary",
    "removes duplicates from a list", "validates an email address",
    "computes the nth Fibonacci number", "parses a CSV file", "sorts a list of tuples",
    "detects a cycle in a linked list", "implements a binary search",
    "counts word frequency in a text", "converts a string to title case",
    "checks if two strings are anagrams",
]
QUALITY_ASPECTS = ["readability", "performance", "memory usage", "error handling", "testability"]
SYSTEM_TYPES = ["microservices", "event-driven", "serverless", "monolithic", "multi-tenant"]
SYSTEM_TASKS = [
    "a real-time chat application", "a payment processing pipeline",
    "an inventory management system", "a video streaming platform",
    "a recommendation engine", "a ride-sharing dispatch system",
    "a multi-region data replication service", "an online auction platform",
]
TECH_STACKS = ["Kafka and PostgreSQL", "Redis and gRPC", "AWS Lambda and DynamoDB", "Kubernetes and RabbitMQ"]
ALGORITHMS = ["quicksort", "merge sort", "a hash table lookup", "depth-first search", "Dijkstra's algorithm", "binary search"]

RECIPIENTS = ["a client", "your manager", "a vendor", "a new hire", "the finance team", "a job candidate", "a former colleague", "the legal team", "an investor", "a supplier"]
TOPICS = [
    "a project deadline extension", "Q3 budget approval", "a contract renewal",
    "an upcoming product launch", "a scheduling conflict", "a pricing change",
    "an internal policy update", "a partnership proposal", "onboarding logistics",
    "a data privacy update", "an office relocation", "a new expense policy",
    "a delayed shipment", "a security incident", "a change in vendor terms",
]
DOC_TYPES = ["memo", "meeting agenda", "project status update", "performance review summary", "press release draft", "onboarding checklist", "risk assessment summary", "vendor comparison brief"]
BUSINESS_CONTEXTS = [
    "a Q3 all-hands meeting", "a new client onboarding", "an internal audit",
    "a product launch", "a vendor negotiation", "a team restructuring",
    "a compliance review", "a supplier evaluation", "a company town hall",
]

METRICS = ["monthly active users", "churn rate", "average order value", "conversion rate", "customer lifetime value", "net promoter score", "daily active users", "gross margin"]
DATA_CONTEXTS = [
    "revenue grew 12% quarter over quarter", "signups doubled in the last month",
    "refund requests rose 8% after the last release", "support tickets dropped 15% after the redesign",
    "customer support response times improved by 20% last month", "cart abandonment rate increased 10% after the checkout redesign",
    "trial-to-paid conversion dropped 5% this quarter", "mobile app downloads tripled after the ad campaign",
]
PERIODS = ["the last 6 months", "the last fiscal year", "Q1 and Q2", "the past 3 quarters", "the last 12 months", "the trailing 4 quarters"]
BUSINESS_QUESTIONS = [
    "the top 10 customers by total spend", "users who churned in the last 30 days",
    "orders placed but never fulfilled", "average session duration by device type",
    "products with the highest return rate", "users who upgraded their plan last quarter",
    "the five most common support ticket categories", "customers who haven't logged in for 60 days",
]

COUNTRIES = ["Portugal", "Vietnam", "Kenya", "Chile", "Finland", "Morocco", "New Zealand", "Mongolia", "Peru", "Iceland",
             "Egypt", "Argentina", "Thailand", "Poland", "Nigeria", "Greece", "Norway", "Philippines", "Ecuador", "Croatia",
             "Jordan", "Uruguay", "Latvia", "Cambodia", "Senegal"]
LANGUAGES_HUMAN = ["French", "Mandarin", "Swahili", "Portuguese", "Korean", "Arabic", "Hindi", "Dutch",
                    "Japanese", "German", "Italian", "Russian", "Turkish", "Polish", "Thai", "Greek"]
PHRASES = ["good morning", "where is the train station", "thank you very much", "I would like a coffee", "see you tomorrow",
           "how much does this cost", "can you help me please", "I am lost", "what time does it open", "happy birthday",
           "I don't understand", "where is the nearest hospital", "this is delicious", "see you soon", "have a safe trip"]
INVENTIONS = ["the telephone", "the printing press", "penicillin", "the World Wide Web", "the light bulb",
              "the airplane", "the steam engine", "the internet search engine", "vaccination", "the microscope",
              "the automobile", "the computer mouse", "radar", "the elevator", "photography"]
EVENTS = ["the first moon landing", "the fall of the Berlin Wall", "the invention of the internet", "the first Olympic Games",
          "the signing of the Treaty of Versailles", "the launch of the first satellite", "the founding of the United Nations",
          "the invention of the personal computer", "the first successful heart transplant", "the collapse of the Soviet Union",
          "the first email ever sent", "the discovery of penicillin", "the first web browser release",
          "the opening of the Suez Canal", "the first commercial flight"]

THEMES = ["a lighthouse keeper who never sees the sea", "a robot learning to paint", "a city that floats", "a librarian who can hear books think",
          "an astronaut who forgot how to feel gravity", "a chef who cooks memories instead of meals", "a mapmaker charting a country that keeps moving",
          "a clockmaker who fell out of time", "a translator who speaks the language of rivers", "a gardener growing forgotten words"]
PRODUCTS = ["a wireless keyboard", "a standing desk", "a noise-cancelling headset", "a smart thermostat", "a coffee grinder",
            "a portable espresso maker", "a fitness tracking ring", "a folding electric bike", "a smart pet feeder", "a travel adapter with USB-C"]
PROBLEMS = ["and it won't turn on", "and the app keeps crashing", "and it stopped syncing yesterday", "and the battery drains in an hour",
            "and the screen flickers constantly", "and it disconnects from Wi-Fi randomly", "and the buttons stopped responding", "and it overheats after 10 minutes"]
EVENT_TYPES = ["a product launch party", "a team offsite", "a customer appreciation dinner", "a conference booth",
               "a quarterly town hall", "a customer advisory board dinner", "an employee wellness day", "a partner appreciation event"]
AUDIENCES = ["40 remote employees", "200 conference attendees", "a group of enterprise clients", "15 new hires",
             "500 trade show visitors", "a board of 12 investors", "30 summer interns", "a regional sales team of 25"]
BUDGETS = ["$2,000", "$15,000", "$500", "$50,000", "$1,200", "$8,500", "$120,000", "$300"]

DEFINE_WORDS = ["latency", "idempotency", "throughput", "amortization", "quorum", "entropy",
                "eventual consistency", "backpressure", "sharding", "checksum", "garbage collection",
                "rate limiting", "denormalization", "observability"]
LIST_TOPICS = ["remote work", "microservices", "open floor plans", "four-day work weeks", "monorepos",
               "standing meetings", "unlimited PTO policies", "pair programming", "feature flags",
               "annual performance reviews", "on-call rotations", "trunk-based development"]
WALKTHROUGH_TASKS = ["set up a home Wi-Fi network", "plan a cross-country move", "negotiate a raise",
                      "onboard a new employee remotely", "prepare for a technical interview",
                      "set up a personal budget", "plan a two-week trip abroad",
                      "migrate a database with zero downtime", "run a retrospective after a failed launch",
                      "choose between two competing job offers"]
VERBOSE_SENTENCES = [
    "Due to the fact that we are currently experiencing a high volume of requests at this point in time, please be advised that response times may be delayed.",
    "In light of the fact that the project timeline has been subject to a number of unforeseen delays, we would like to take this opportunity to inform all stakeholders accordingly.",
    "It has come to our attention that, on account of recent changes to the underlying infrastructure, certain features may not be functioning in the manner in which they were originally intended.",
    "At this juncture in time, and taking into consideration the various constraints that have been placed upon the team, it would appear that a revised approach may be warranted.",
    "For the purposes of clarity and in order to avoid any potential misunderstanding, we felt it necessary to reiterate the policy in its entirety.",
    "Owing to the circumstances that have arisen as a direct result of the recent system outage, a number of customers have experienced a disruption in service.",
]


def dedupe(prompts: list[str]) -> list[str]:
    seen = set()
    unique = []
    for p in prompts:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


def build(target: int, templates: list) -> list[str]:
    """Oversamples generously (10x target) then dedupes down to the target,
    since random.choice-based sampling hits collisions well before every
    combinatorial slot is exhausted."""
    samples = [random.choice(templates)() for _ in range(target * 10)]
    return dedupe(samples)[:target]


def build_coding(n: int) -> list[str]:
    templates = [
        lambda: f"Write a {random.choice(LANGUAGES)} function that {random.choice(TASKS)}.",
        lambda: f"Explain the difference between {random.choice(CONCEPTS_A)} and {random.choice(CONCEPTS_B)} in {random.choice(LANGUAGES)}.",
        lambda: f"Refactor a {random.choice(LANGUAGES)} function that {random.choice(TASKS)} to improve {random.choice(QUALITY_ASPECTS)}.",
        lambda: f"Design a {random.choice(SYSTEM_TYPES)} architecture for {random.choice(SYSTEM_TASKS)} using {random.choice(TECH_STACKS)}, covering fault tolerance, data consistency, and scaling under peak load.",
        lambda: f"What is the time complexity of {random.choice(ALGORITHMS)}, and when would you choose it over alternatives?",
        lambda: f"Write unit tests in {random.choice(LANGUAGES)} for a function that {random.choice(TASKS)}.",
        lambda: f"Debug this issue: a {random.choice(LANGUAGES)} function that {random.choice(TASKS)} works for most inputs but fails on edge cases. What should I check?",
        lambda: f"Compare {random.choice(CONCEPTS_A)} and {random.choice(CONCEPTS_B)} for a system handling {random.choice(SYSTEM_TASKS)}.",
    ]
    return build(n, templates)


def build_office(n: int) -> list[str]:
    templates = [
        lambda: f"Write a professional email to {random.choice(RECIPIENTS)} about {random.choice(TOPICS)}.",
        lambda: f"Draft a {random.choice(DOC_TYPES)} for {random.choice(BUSINESS_CONTEXTS)}.",
        lambda: f"Write a polite follow-up message to {random.choice(RECIPIENTS)} regarding {random.choice(TOPICS)}.",
        lambda: f"Summarize the key points from a discussion about {random.choice(TOPICS)} into three action items.",
    ]
    return build(n, templates)


def build_data(n: int) -> list[str]:
    templates = [
        lambda: f"Given that {random.choice(DATA_CONTEXTS)}, what might be driving the change in {random.choice(METRICS)}?",
        lambda: f"What SQL query would return {random.choice(BUSINESS_QUESTIONS)}?",
        lambda: f"Analyze the trend in {random.choice(METRICS)} over {random.choice(PERIODS)} and suggest two hypotheses for what's driving it.",
    ]
    return build(n, templates)


def build_general(n: int) -> list[str]:
    templates = [
        lambda: f"What is the capital of {random.choice(COUNTRIES)}?",
        lambda: f"Translate '{random.choice(PHRASES)}' into {random.choice(LANGUAGES_HUMAN)}.",
        lambda: f"Who invented {random.choice(INVENTIONS)}?",
        lambda: f"What year did {random.choice(EVENTS)} happen?",
    ]
    return build(n, templates)


def build_creative_support(n: int) -> list[str]:
    templates = [
        lambda: f"Write a short story about {random.choice(THEMES)}.",
        lambda: f"A customer says their {random.choice(PRODUCTS)} isn't working {random.choice(PROBLEMS)}. Write a helpful support response.",
        lambda: f"Plan {random.choice(EVENT_TYPES)} for {random.choice(AUDIENCES)} with a budget of {random.choice(BUDGETS)}, including a rough timeline and vendor checklist.",
        lambda: f"Write a product description for {random.choice(PRODUCTS)} aimed at first-time buyers.",
    ]
    return build(n, templates)


def build_misc(n: int) -> list[str]:
    templates = [
        lambda: f"What is {random.randint(2, 999)} plus {random.randint(2, 999)}?",
        lambda: f"Define '{random.choice(DEFINE_WORDS)}' in one sentence.",
        lambda: f"Rewrite this sentence more concisely: '{random.choice(VERBOSE_SENTENCES)}'",
        lambda: f"List three pros and three cons of {random.choice(LIST_TOPICS)}.",
        lambda: f"Walk me through the steps to {random.choice(WALKTHROUGH_TASKS)}, covering the key decisions at each stage.",
    ]
    return build(n, templates)


def main():
    rows = []
    rows += [("coding_technical", p) for p in build_coding(350)]
    rows += [("office_business", p) for p in build_office(150)]
    rows += [("data_analysis", p) for p in build_data(100)]
    rows += [("general_knowledge", p) for p in build_general(100)]
    rows += [("creative_support", p) for p in build_creative_support(150)]
    rows += [("misc", p) for p in build_misc(150)]

    random.shuffle(rows)

    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["category", "prompt"])
        writer.writerows(rows)

    print(f"Wrote {len(rows)} prompts to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
