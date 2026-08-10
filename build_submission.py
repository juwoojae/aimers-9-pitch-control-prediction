"""공식 베이스라인과 같은 Unix 권한을 가진 ``submit.zip``을 생성한다."""

from __future__ import annotations

import argparse
import stat
import zipfile
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
SUBMISSION_FILES = (Path("model/rf.pkl"), Path("script.py"), Path("requirements.txt"))

# 재실행할 때 파일 내용이 같으면 ZIP도 같아지도록 타임스탬프를 고정한다.
FIXED_TIMESTAMP = (2026, 8, 9, 0, 0, 0)


def directory_info(name: str) -> zipfile.ZipInfo:
    """ZIP 안의 디렉터리에 Unix 0755 권한을 지정한다."""
    info = zipfile.ZipInfo(name, FIXED_TIMESTAMP)
    info.create_system = 3
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = (stat.S_IFDIR | 0o755) << 16
    return info


def file_info(name: str) -> zipfile.ZipInfo:
    """ZIP 안의 일반 파일에 Unix 0644 권한을 지정한다."""
    info = zipfile.ZipInfo(name, FIXED_TIMESTAMP)
    info.create_system = 3
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    return info


def build_archive(output: Path) -> None:
    missing = [
        str(relative_path)
        for relative_path in SUBMISSION_FILES
        if not (ROOT_DIR / relative_path).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"제출 ZIP에 필요한 파일이 없음: {missing}")

    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        output, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        # 일부 평가기가 명시적인 model/ 엔트리를 요구하므로 먼저 추가한다.
        archive.writestr(directory_info("model/"), b"")
        for relative_path in SUBMISSION_FILES:
            archive.writestr(
                file_info(relative_path.as_posix()),
                (ROOT_DIR / relative_path).read_bytes(),
            )
    print(f"제출 ZIP 생성: {output.resolve()} ({output.stat().st_size} bytes)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT_DIR / "submit.zip"
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    build_archive(arguments.output)
