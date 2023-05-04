from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from third_parties.linkedin import scrape_linkedin_profile
from agents.linkedin_lookup_agent import lookup


if __name__ == "__main__":
    print("langchain")

    linkedin_profile_url = lookup(name="Eden Marco")

    summary_template = """
        given the linkedin information {information} about a person from I want to create:
        1. a short summary
        2. two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    # temperature determines how creative the model is going to be, 0 not too creative
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    # chain and run the llm in a chain
    chain = LLMChain(llm=llm, prompt=summary_prompt_template)

    linkedin_data = scrape_linkedin_profile(linkedin_profile_url=linkedin_profile_url)

    print(chain.run(information=linkedin_data))
