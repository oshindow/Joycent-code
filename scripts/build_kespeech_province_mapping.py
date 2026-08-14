#!/usr/bin/env python3
"""Add province/municipality labels to the official KeSpeech city metadata."""

import argparse
from collections import Counter
from pathlib import Path


CITY_TO_PROVINCE = {
    "Baoding": "Hebei",
    "Beijing": "Beijing",
    "Changchun": "Jilin",
    "Chaoyang": "Liaoning",
    "Chengde": "Hebei",
    "Chengdu": "Sichuan",
    "Chifeng": "InnerMongolia",
    "Chongqing": "Chongqing",
    "Dalian": "Liaoning",
    "Dandong": "Liaoning",
    "Guilin": "Guangxi",
    "Guiyang": "Guizhou",
    "Harbin": "Heilongjiang",
    "Hefei": "Anhui",
    "Jinan": "Shandong",
    "Kunming": "Yunnan",
    "Langfang": "Hebei",
    "Lanzhou": "Gansu",
    "Nanjing": "Jiangsu",
    "Qingdao": "Shandong",
    "Shenyang": "Liaoning",
    "Shijiazhuang": "Hebei",
    "Shizuishan": "Ningxia",
    "Tangshan": "Hebei",
    "Weihai": "Shandong",
    "Wuzhong": "Ningxia",
    "XiAn": "Shaanxi",
    "Xingtai": "Hebei",
    "Yangzhou": "Jiangsu",
    "Yantai": "Shandong",
    "Yinchuan": "Ningxia",
    "Yingkou": "Liaoning",
    "Zhengzhou": "Henan",
    "Zhongwei": "Ningxia",
}

PROVINCE_CHINESE = {
    "Anhui": "安徽省", "Beijing": "北京市", "Chongqing": "重庆市",
    "Gansu": "甘肃省", "Guangxi": "广西壮族自治区", "Guizhou": "贵州省",
    "Hebei": "河北省", "Heilongjiang": "黑龙江省", "Henan": "河南省",
    "InnerMongolia": "内蒙古自治区", "Jiangsu": "江苏省", "Jilin": "吉林省",
    "Liaoning": "辽宁省", "Ningxia": "宁夏回族自治区", "Shaanxi": "陕西省",
    "Shandong": "山东省", "Sichuan": "四川省", "Yunnan": "云南省",
}


def read_map(path: Path) -> dict[str, str]:
    result = {}
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = raw.split()
        if len(fields) != 2:
            raise ValueError(f"{path}:{number}: expected two fields")
        key, value = fields
        if key in result:
            raise ValueError(f"{path}:{number}: duplicate key {key}")
        result[key] = value
    return result


def write_rows(path: Path, rows) -> None:
    path.write_text("".join("|".join(row) + "\n" for row in rows), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=Path("/data1/xintong/datasets/kespeech/KeSpeech/Metadata"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("resources/multi_accent/kespeech"),
    )
    args = parser.parse_args()

    city_to_chinese = read_map(args.metadata_dir / "city2Chinese")
    city_to_subdialect = read_map(args.metadata_dir / "city2subdialect")
    subdialect_to_chinese = read_map(args.metadata_dir / "subdialect2Chinese")
    speaker_to_city = read_map(args.metadata_dir / "spk2city")

    official_cities = set(city_to_chinese) | set(city_to_subdialect) | set(speaker_to_city.values())
    if set(CITY_TO_PROVINCE) != official_cities:
        raise RuntimeError(
            f"province mapping mismatch; missing={sorted(official_cities - set(CITY_TO_PROVINCE))}, "
            f"extra={sorted(set(CITY_TO_PROVINCE) - official_cities)}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    city_rows = sorted(
        (
            (city, CITY_TO_PROVINCE[city], city_to_subdialect[city])
            for city in official_cities
        ),
        key=lambda row: (row[1], row[0]),
    )
    write_rows(args.output_dir / "city_province_subdialect.txt", city_rows)

    speaker_province_rows = []
    speaker_subdialect_rows = []
    province_counts = Counter()
    subdialect_counts = Counter()
    for speaker, city in sorted(speaker_to_city.items()):
        province = CITY_TO_PROVINCE[city]
        subdialect = city_to_subdialect[city]
        speaker_province_rows.append((speaker, province))
        speaker_subdialect_rows.append((speaker, subdialect))
        province_counts[province] += 1
        subdialect_counts[subdialect] += 1
    write_rows(args.output_dir / "spk2province.txt", speaker_province_rows)
    write_rows(args.output_dir / "spk2regional_subdialect.txt", speaker_subdialect_rows)

    lines = [
        "# KeSpeech province mapping",
        "",
        "Source metadata:",
        "",
        f"- `{args.metadata_dir / 'spk2city'}`",
        f"- `{args.metadata_dir / 'city2Chinese'}`",
        f"- `{args.metadata_dir / 'city2subdialect'}`",
        f"- `{args.metadata_dir / 'subdialect2Chinese'}`",
        "",
        "`city_province_subdialect.txt` is a headerless, pipe-separated `city|province_or_municipality|regional_subdialect` mapping.",
        "",
        "`spk2province.txt` and `spk2regional_subdialect.txt` are derived by joining the official `spk2city` with the city mapping.",
        "",
        "Important: `regional_subdialect` is the city-level dialect-region prior from official `city2subdialect`; it is not the utterance-level label in `phase1.utt2subdialect`, which may be `Mandarin` or a regional subdialect for different utterances from the same speaker.",
        "",
        "## City mapping",
        "",
        "| City | City (Chinese) | Province/municipality | Chinese | Regional subdialect | Chinese |",
        "|---|---|---|---|---|---|",
    ]
    for city, province, subdialect in city_rows:
        lines.append(
            f"| {city} | {city_to_chinese[city]} | {province} | {PROVINCE_CHINESE[province]} | "
            f"{subdialect} | {subdialect_to_chinese[subdialect]} |"
        )
    lines += [
        "",
        "## Speakers per province/municipality",
        "",
        "| Province/municipality | Chinese | Speakers |",
        "|---|---|---:|",
    ]
    for province, count in sorted(province_counts.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| {province} | {PROVINCE_CHINESE[province]} | {count} |")
    lines += [
        "",
        "## Speakers per city-level regional subdialect",
        "",
        "| Regional subdialect | Chinese | Speakers |",
        "|---|---|---:|",
    ]
    for subdialect, count in sorted(subdialect_counts.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| {subdialect} | {subdialect_to_chinese[subdialect]} | {count} |")
    lines += [
        "",
        f"Total: **{len(official_cities)} cities, {len(province_counts)} province/municipality labels, "
        f"{len(subdialect_counts)} regional subdialects, and {len(speaker_to_city)} speakers**.",
    ]
    (args.output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"wrote {len(city_rows)} city mappings")
    print(f"wrote {len(speaker_to_city)} speaker mappings")
    print(f"output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
