from gpt_researcher import GPTResearcher
from dotenv import load_dotenv
load_dotenv()

def researcher_config(): # read https://docs.gptr.dev/docs/gpt-researcher/gptr/config for config options and more details
    import os
    os.environ["FAST_LLM"] = "google_genai:gemini-2.0-flash"
    os.environ["SMART_LLM"] = "google_genai:gemini-2.0-flash"
    os.environ["STRATEGIC_LLM"] = "google_genai:gemini-2.0-flash"
    os.environ["EMBEDDING"] = "google_genai:models/text-embedding-004"

researcher_config()


async def conduct_research(query, deep_research: bool = False, report_source: str = "web") -> str:
    """
    This function conducts research on the given query and returns the report.

    This uses the GPTResearcher which conduct deep research on the given query.

    This takes some time to complete.
    """
    # Report Type
    report_type = "deep" if deep_research else "research_report" 

    # Initialize the researcher
    researcher = GPTResearcher(query=query, report_type=report_type, report_source=report_source)
    # Conduct research on the given query
    await researcher.conduct_research()
    # Write the report
    report = await researcher.write_report()
    return report