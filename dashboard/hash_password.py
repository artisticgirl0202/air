from __future__ import annotations

import argparse
import getpass
import sys

import bcrypt


def hash_password(plain: str, *, rounds: int = 12) -> str:
    """Return a bcrypt hash for config.yaml. Never store the plaintext password."""
    if not isinstance(plain, str) or not plain:
        raise ValueError("비밀번호는 비어 있을 수 없습니다.")
    if rounds < 12:
        raise ValueError("bcrypt cost는 12 이상이어야 합니다.")
    digest = bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt(rounds=rounds))
    return digest.decode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="bcrypt 비밀번호 해시 생성")
    parser.add_argument(
        "--password",
        help="평문 비밀번호. 생략하면 화면에 표시되지 않게 입력받습니다.",
    )
    parser.add_argument("--rounds", type=int, default=12)
    args = parser.parse_args()

    plain = args.password if args.password is not None else getpass.getpass("평문 비밀번호: ")
    hashed = hash_password(plain, rounds=args.rounds)
    print(hashed)
    print("위 값을 dashboard/config.yaml 의 password 필드에 붙여 넣으세요.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
