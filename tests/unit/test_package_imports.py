import subprocess
import sys


def test_shipping_package_imports_cleanly() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            "import arkive_rag, arkive_rag.contracts; "
            "print(arkive_rag.__name__, arkive_rag.contracts.__name__, sep='\\n')",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "arkive_rag",
        "arkive_rag.contracts",
    ]
