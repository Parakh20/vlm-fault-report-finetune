BENCHMARK_TASKS: list[dict] = [
    # --- Category 1: Information Extraction ---
    {
        "id": "ie_01", "category": "information_extraction",
        "prompt": "Find the current price of Reliance Industries stock on NSE and return the price, change %, and 52-week high",
    },
    {
        "id": "ie_02", "category": "information_extraction",
        "prompt": "Go to arxiv.org and find the 3 most recent papers on 'retrieval augmented generation', return titles + abstracts",
    },
    {
        "id": "ie_03", "category": "information_extraction",
        "prompt": "Find the weather forecast for Mumbai for the next 5 days",
    },
    {
        "id": "ie_04", "category": "information_extraction",
        "prompt": "Find the top 5 results for 'best budget laptops India 2026' and extract product names and prices",
    },
    {
        "id": "ie_05", "category": "information_extraction",
        "prompt": "Go to IITB.ac.in and find the academic calendar for the current semester",
    },
    # --- Category 2: Multi-step Navigation ---
    {
        "id": "nav_01", "category": "multi_step_navigation",
        "prompt": "Go to HackerNews, find today's #1 story, open it, and summarize the article in 3 bullet points",
    },
    {
        "id": "nav_02", "category": "multi_step_navigation",
        "prompt": "Search for 'YC S25 batch' on Google, find the YC company list page, and return the first 10 company names and their one-line descriptions",
    },
    {
        "id": "nav_03", "category": "multi_step_navigation",
        "prompt": "Go to GitHub, search for 'FastMCP python', open the top result, and return the README summary and star count",
    },
    {
        "id": "nav_04", "category": "multi_step_navigation",
        "prompt": "Find the Wikipedia page for 'Limit order book', navigate to the 'Market microstructure' linked article, and return its first paragraph",
    },
    {
        "id": "nav_05", "category": "multi_step_navigation",
        "prompt": "Go to producthunt.com, find today's #1 product, and return its name, tagline, and upvote count",
    },
    # --- Category 3: Form Interaction ---
    {
        "id": "form_01", "category": "form_interaction",
        "prompt": "Go to DuckDuckGo, search for 'IIT Bombay CSE faculty', then refine the search to show only recent results from the last year",
    },
    {
        "id": "form_02", "category": "form_interaction",
        "prompt": "Go to Google Flights (no login), search for flights from Mumbai to Delhi tomorrow, return the cheapest 3 options with times and prices",
    },
    {
        "id": "form_03", "category": "form_interaction",
        "prompt": "Go to Wolfram Alpha and compute the integral of x^2 * sin(x) from 0 to pi. Return the result.",
    },
    {
        "id": "form_04", "category": "form_interaction",
        "prompt": "Use Google Scholar to find papers by 'Sunita Sarawagi' and return her h-index",
    },
    {
        "id": "form_05", "category": "form_interaction",
        "prompt": "Go to translate.google.com, translate 'The quick brown fox jumps over the lazy dog' to Hindi and return the translation",
    },
    # --- Category 4: Data Aggregation ---
    {
        "id": "agg_01", "category": "data_aggregation",
        "prompt": "Find the top 10 trending repositories on GitHub today, return name, stars, and description for each",
    },
    {
        "id": "agg_02", "category": "data_aggregation",
        "prompt": "Go to NSE India website, find the top 5 gainers and top 5 losers for today, return the full table",
    },
    {
        "id": "agg_03", "category": "data_aggregation",
        "prompt": "Find the current USD to INR, EUR to INR, and GBP to INR exchange rates from a financial site",
    },
    {
        "id": "agg_04", "category": "data_aggregation",
        "prompt": "Go to Cricbuzz, find the scorecard of the most recent completed international match, return full scores",
    },
    {
        "id": "agg_05", "category": "data_aggregation",
        "prompt": "Find the 5 most upvoted questions tagged 'python' on StackOverflow from the last week",
    },
    # --- Category 5: Reasoning Over Web Content ---
    {
        "id": "reason_01", "category": "reasoning_over_content",
        "prompt": "Search for 'India drone regulations 2025', read the top 3 results, and summarize the key rules in bullet points",
    },
    {
        "id": "reason_02", "category": "reasoning_over_content",
        "prompt": "Find recent news about Anthropic from the last month, read 3 articles, and identify the 3 biggest developments",
    },
    {
        "id": "reason_03", "category": "reasoning_over_content",
        "prompt": "Search for 'best MCP servers 2025', compile a list of the top 10 recommended servers with their use cases",
    },
    {
        "id": "reason_04", "category": "reasoning_over_content",
        "prompt": "Find the latest SEBI circular on F&O regulations, summarize what changed and who it affects",
    },
    {
        "id": "reason_05", "category": "reasoning_over_content",
        "prompt": "Research 'options trading strategies for beginners', read 2-3 sources, and produce a structured comparison of covered calls vs cash-secured puts",
    },
]

if __name__ == "__main__":
    for t in BENCHMARK_TASKS:
        print(t["id"], "-", t["category"])
