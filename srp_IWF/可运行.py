import requests
from bs4 import BeautifulSoup
import pandas as pd


def scrape_iwf_cards(url, output_file="iwf_results.xlsx"):
    """
    爬取 IWF 网站的 card 样式举重数据
    """
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
    cards = soup.find_all('div', class_='card')

    if not cards:
        print("未找到任何 card 元素，请检查网页结构或尝试使用 Selenium")
        return None

    data = []
    last_title = ""

    for card in cards:
        try:
            # 跳过表头
            if "card__legend" in card.get("class", []):
                continue

            # 查找该卡片对应的项目标题（向上查找最近的 <h2>）
            title_tag = card.find_previous("h2")
            title = title_tag.get_text(strip=True) if title_tag else last_title
            last_title = title  # 缓存当前标题，方便后续同组卡片使用

            # 获取选手姓名（兼容 a.title 或普通 p）
            a_tag = card.find('a', class_='title')
            if a_tag and a_tag.find('p'):
                name = a_tag.find('p').get_text(strip=True)
            else:
                # 如果没有 a.title，则直接取 card 内第一个 Name 字段
                name_tag = card.find('p', string=lambda s: s and 'Name' not in s)
                name = name_tag.get_text(strip=True) if name_tag else ''

            link = a_tag['href'] if a_tag and a_tag.has_attr('href') else ''

            # 获取 rank
            rank_tag = card.find('span', class_='only__mobile', string=lambda s: 'Rank' in s if s else False)
            rank = rank_tag.find_parent('p').get_text(strip=True).replace('Rank:', '') if rank_tag else ''

            # 获取国家代码
            flag_tag = card.find('span', class_='flag')
            nation = flag_tag.find_next_sibling(text=True).strip() if flag_tag else ''

            # 出生日期、体重、组别
            born_tag = card.find('span', string=lambda s: 'Born' in s if s else False)
            born = born_tag.find_parent('p').get_text(strip=True).replace('Born:', '') if born_tag else ''

            bw_tag = card.find('span', string=lambda s: 'B.weight' in s if s else False)
            body_weight = bw_tag.find_parent('p').get_text(strip=True).replace('B.weight:', '') if bw_tag else ''

            group_tag = card.find('span', string=lambda s: 'Group' in s if s else False)
            group = group_tag.find_parent('p').get_text(strip=True).replace('Group:', '') if group_tag else ''

            # 抓举、挺举、总成绩
            snatch_tag = card.find('span', string=lambda s: 'Snatch' in s if s else False)
            snatch = snatch_tag.find_parent('p').find('strong').get_text(strip=True) if snatch_tag else ''

            cj_tag = card.find('span', string=lambda s: 'CI' in s if s else False)
            clean_jerk = cj_tag.find_parent('p').find('strong').get_text(strip=True) if cj_tag else ''

            total_tag = card.find('span', string=lambda s: 'Total' in s if s else False)
            total = total_tag.find_parent('p').get_text(strip=True).replace('Total:', '') if total_tag else ''

            data.append({
                "Event": title,
                "Rank": rank,
                "Name": name,
                "Nation": nation,
                "Born": born,
                "Body_Weight": body_weight,
                "Group": group,
                "Snatch": snatch,
                "Clean_Jerk": clean_jerk,
                "Total": total,
                # "Profile_Link": link
            })

        except Exception as e:
            print(f"解析单个 card 时出错: {e}")
    
    # 在不同 Event 之间插入空行，增加可读性
    spaced_data = []
    last_event = None
    for row in data:
        if last_event is not None and row["Event"] != last_event:
            spaced_data.append({key: "" for key in row.keys()})  # 插入空行
        spaced_data.append(row)
        last_event = row["Event"]
    data = spaced_data

    df = pd.DataFrame(data)
    print(f"共提取到 {len(df)} 条记录。")
    print(df.head())

    df.to_excel(output_file, index=False, engine='openpyxl')
    print(f"数据已保存到 {output_file}")

    return df


# 测试
if __name__ == "__main__":
    url = "https://iwf.sport/results/results-by-events/results-by-events-2018-2025/?event_id=577"
    scrape_iwf_cards(url)
