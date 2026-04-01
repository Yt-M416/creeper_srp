'''
从 IWF 官网爬取举重世锦赛数据
'''

import requests
from bs4 import BeautifulSoup
import pandas as pd
import os

def scrape_iwf_cards(url):
    headers = {
        'User-Agent': (
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/123.0.0.0 Safari/537.36'
        )
    }

    print(f"正在访问网页: {url}")
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    response.encoding = 'utf-8'

    soup = BeautifulSoup(response.text, 'html.parser')

    h2_tag = soup.find('h2')
    competition_name = h2_tag.get_text(strip=True) if h2_tag else "IWF Competition"
    print(f"比赛名称: {competition_name}")

    cards = soup.find_all('div', class_='card')
    if not cards:
        print("未找到任何 card 元素，请检查网页结构或尝试使用 Selenium")
        return None, competition_name

    data = []
    last_title = ""
    current_event = None
    rank1_seen = False
    collecting = True

    for card in cards:
        try:
            if card.find('p', string='Snatch'):
                print("遇到 Snatch，停止抓取后续重复数据")
                break

            if "card__legend" in card.get("class", []):
                continue

            title_tag = card.find_previous("h2")
            title = title_tag.get_text(strip=True) if title_tag else last_title

            if title != last_title:
                current_event = title
                rank1_seen = False
                collecting = True

            last_title = title

            rank_tag = card.find('span', class_='only__mobile', string=lambda s: 'Rank' in s if s else False)
            rank = rank_tag.find_parent('p').get_text(strip=True).replace('Rank:', '') if rank_tag else ''

            if rank == '1':
                if not rank1_seen:
                    rank1_seen = True
                else:
                    collecting = False
                    continue

            if not collecting:
                continue

            a_tag = card.find('a', class_='title')
            name = ''
            if a_tag:
                p_tags = a_tag.find_all('p')
                if len(p_tags) >= 2:
                    name = p_tags[1].get_text(strip=True)
                elif len(p_tags) == 1:
                    name = p_tags[0].get_text(strip=True)

            flag_tag = card.find('span', class_='flag')
            nation = flag_tag.find_next_sibling(text=True).strip() if flag_tag else ''

            born_tag = card.find('span', string=lambda s: 'Born' in s if s else False)
            born = born_tag.find_parent('p').get_text(strip=True).replace('Born:', '') if born_tag else ''

            bw_tag = card.find('span', string=lambda s: 'B.weight' in s if s else False)
            body_weight = bw_tag.find_parent('p').get_text(strip=True).replace('B.weight:', '') if bw_tag else ''

            group_tag = card.find('span', string=lambda s: 'Group' in s if s else False)
            group = group_tag.find_parent('p').get_text(strip=True).replace('Group:', '') if group_tag else ''

            snatch_tag = card.find('span', string=lambda s: 'Snatch' in s if s else False)
            snatch = snatch_tag.find_parent('p').find('strong').get_text(strip=True) if snatch_tag else ''

            cj_tag = card.find('span', string=lambda s: 'CI' in s if s else False)
            clean_jerk = cj_tag.find_parent('p').find('strong').get_text(strip=True) if cj_tag else ''

            total_tag = card.find('span', string=lambda s: 'Total' in s if s else False)
            total = total_tag.find_parent('p').get_text(strip=True).replace('Total:', '') if total_tag else ''

            data.append({
                "Rank": rank,
                "Name": name,
                "Nation": nation,
                "Born": born,
                "Body_Weight": body_weight,
                "Group": group,
                "Snatch": snatch,
                "Clean_Jerk": clean_jerk,
                "Total": total,
                "Event": title,
            })

        except Exception as e:
            print(f"解析单个 card 时出错: {e}")

    spaced_data = []
    last_event = None
    for row in data:
        if last_event is not None and row["Event"] != last_event:
            spaced_data.append({key: "" for key in row.keys()})
        spaced_data.append(row)
        last_event = row["Event"]
    data = spaced_data

    df = pd.DataFrame(data)
    return df, competition_name


def scrape_multiple_iwf_cards(urls, output_file="iwf_1998-2017.xlsx"):
    """
    批量爬取多个 IWF 比赛结果，每个比赛放在 Excel 的不同 sheet。
    如果文件不存在则创建；如果文件存在但不是 Excel 则覆盖创建。
    """
    # 检查文件是否存在
    if not os.path.exists(output_file):
        mode = 'w'
        print(f"{output_file} 不存在，将创建新文件")
    else:
        # 尝试读取，确保是有效 Excel 文件
        try:
            pd.read_excel(output_file)
            mode = 'a'
        except Exception:
            mode = 'w'
            print(f"{output_file} 不是有效 Excel 文件，将重新创建")

    if mode == 'w':
        # 创建新文件时，不使用 if_sheet_exists
        with pd.ExcelWriter(output_file, engine="openpyxl", mode=mode) as writer:
            for url in urls:
                df, comp_name = scrape_iwf_cards(url)
                if df is not None:
                    sheet_name = comp_name[:31]
                    df.to_excel(writer, index=False, sheet_name=sheet_name)
    else:
        # 追加已有文件时，使用 if_sheet_exists="replace"
        with pd.ExcelWriter(output_file, engine="openpyxl", mode=mode, if_sheet_exists="replace") as writer:
            for url in urls:
                df, comp_name = scrape_iwf_cards(url)
                if df is not None:
                    sheet_name = comp_name[:31]
                    df.to_excel(writer, index=False, sheet_name=sheet_name)

    print(f"所有比赛数据已保存到 {output_file}")


# =================== 测试运行 ===================
    # 00，04，08，12，16 没有
if __name__ == "__main__":
    urls = [
        "https://iwf.sport/results/results-by-events/results-by-events-upto2018/?event_id=96",
        "https://iwf.sport/results/results-by-events/results-by-events-upto2018/?event_id=99",
        "https://iwf.sport/results/results-by-events/results-by-events-upto2018/?event_id=89",
        "https://iwf.sport/results/results-by-events/results-by-events-upto2018/?event_id=91",
        "https://iwf.sport/results/results-by-events/results-by-events-upto2018/?event_id=94",
        "https://iwf.sport/results/results-by-events/results-by-events-upto2018/?event_id=3",
        "https://iwf.sport/results/results-by-events/results-by-events-upto2018/?event_id=8",
        "https://iwf.sport/results/results-by-events/results-by-events-upto2018/?event_id=11",
        "https://iwf.sport/results/results-by-events/results-by-events-upto2018/?event_id=41",
        "https://iwf.sport/results/results-by-events/results-by-events-upto2018/?event_id=116",
        "https://iwf.sport/results/results-by-events/results-by-events-upto2018/?event_id=190",
        "https://iwf.sport/results/results-by-events/results-by-events-upto2018/?event_id=253",
        "https://iwf.sport/results/results-by-events/results-by-events-upto2018/?event_id=298",
        "https://iwf.sport/results/results-by-events/results-by-events-upto2018/?event_id=341",
        "https://iwf.sport/results/results-by-events/results-by-events-upto2018/?event_id=405",
        
    ]
    scrape_multiple_iwf_cards(urls)
