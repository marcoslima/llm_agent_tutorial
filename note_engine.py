from pathlib import Path

from llama_index.core.tools import FunctionTool

note_file = Path('data/notes.txt')


def save_note(note):
    if not note_file.exists():
        open(note_file, "w")

    with open(note_file, "a") as f:
        f.writelines([note + "\n"])

    return "note saved"


note_engine = FunctionTool.from_defaults(
    fn=save_note,
    name="note_saver",
    description="this tool can save a text based note to a file for the user",
)
