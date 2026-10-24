import json
import sys


NOTEBOOKS = (
    ("## GRU without k fold", "CNN-GRU"),
    ("lstm_model.run_model", "LSTM"),
)


def main():
    notebook = json.load(sys.stdin)

    for marker, title in NOTEBOOKS:
        for index, cell in enumerate(notebook.get("cells", [])):
            source = "".join(cell.get("source", []))
            if marker in source:
                notebook["cells"] = [
                    {
                        "cell_type": "markdown",
                        "metadata": {},
                        "source": [
                            f"# {title}\n",
                            "\n",
                            "Model implementation intentionally omitted from the public repository.\n",
                        ],
                    },
                    *notebook["cells"][index:],
                ]
                json.dump(notebook, sys.stdout, ensure_ascii=False, indent=1)
                sys.stdout.write("\n")
                return

    raise SystemExit("Could not find the public training/testing section; refusing to expose the notebook")


if __name__ == "__main__":
    main()
