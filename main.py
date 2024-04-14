from dotenv import load_dotenv
import pandas as pd
from pathlib import Path

from llama_index.experimental import PandasQueryEngine
from llama_index.llms.openai import OpenAI

from prompts import new_prompt, instruction_str, context
from note_engine import note_engine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from pdf import brasil_engine


load_dotenv()


def main():
    population_path = Path('data/WorldPopulation2023.csv')
    population_df = pd.read_csv(population_path)

    population_query_engine = PandasQueryEngine(
        df=population_df,
        verbose=True,
        instruction_str=instruction_str,
    )
    population_query_engine.update_prompts({'pandas_prompt': new_prompt})
    # population_query_engine.query("what is the population of brazil?")

    tools = [
        note_engine,
        QueryEngineTool(
            query_engine=population_query_engine,
            metadata=ToolMetadata(
                name='Population',
                description='A query engine for world population statistics.',
            ),
        ),
        QueryEngineTool(
            query_engine=brasil_engine,
            metadata=ToolMetadata(
                name='Brasil',
                description='this gives detailed information about Brasil, '
                            'the country.',
            ),
        ),
    ]

    llm = OpenAI(model='gpt-3.5-turbo-0613')
    agent = ReActAgent.from_tools(tools,
                                  llm=llm,
                                  verbose=True,
                                  context=context)

    while (prompt := input('Enter a prompt (q to quit): ')) != 'q':
        result = agent.query(prompt)
        print(result)


if __name__ == '__main__':
    main()
