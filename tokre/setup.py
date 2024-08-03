from pathlib import Path
from typing import Optional

import tokre
from tokre.utils import hash_tokenizer


def setup(
    tokenizer, workspace: Optional[str] = None, openai_api_key: Optional[str] = None
):
    """
    Sets up tokre.

    Before being able to use tokre, you must provide a tokenizer class that implements `encode` and `decode`.

    If workspace is provided, it's used as a directory to load and store scripts.
    If the workspace doesn't exist in the current directory, a directory is created.

    """
    assert tokenizer is not None, "`tokenizer` needs to be provided."
    if workspace is not None:
        assert workspace != "", "`workspace` can't be an empty string."

    if openai_api_key is not None:
        tokre._openai_api_key = openai_api_key

    if workspace is None:
        if tokre._workspace is not None:
            assert (
                False
            ), "Cannot set only `tokenizer` using tokre.setup when `workspace` is already set. Please re-run tokre.setup with both a `tokenizer` and `workspace` argument."
        tokre._tokenizer = tokenizer

    else:
        # (if workspace is not None)

        ws_path = Path(workspace)

        # If the workspace doesn't exist, create it with .tokenizer_hash
        if not ws_path.is_dir():
            ws_path.mkdir(parents=True)

            # Write a hash of the tokenizer to .tokenizer_hash
            with open(ws_path / ".tokenizer_hash", "w") as f:
                f.write(hash_tokenizer(tokenizer))

        # If the workspace exists, make sure the tokenizer matches
        else:
            tokenizer_hash_path = ws_path / ".tokenizer_hash"

            # make sure .tokenizer_hash is inside the workspace
            assert tokenizer_hash_path.exists(), (
                f".tokenizer_hash wasn't found inside {ws_path}.\n"
                "Pre-existing workspace should have a tokenizer hash stored. Aborting workspace setup."
            )

            # make sure .tokenizer_hash matches the provided tokenizer's hash
            assert tokenizer_hash_path.read_text() == hash_tokenizer(tokenizer), (
                f"Hash of the provided tokenizer doesn't match the hash stored in {tokenizer_hash_path}\n",
                f"Aborting workspace setup.",
            )

        ws_path = ws_path.resolve()  # resolve makes it an absolute path

        tokre._workspace = ws_path
        tokre._tokenizer = tokenizer
