from dotenv import load_dotenv
import pandas as pd
from pathlib import Path

from llama_index.experimental import PandasQueryEngine
from prompts import new_prompt, instruction_str

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
    population_query_engine.query("what is the population of brazil?")


if __name__ == '__main__':
    main()
