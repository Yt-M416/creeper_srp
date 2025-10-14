import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import time
import re

def setup_driver():

    '''
    """配置Chrome浏览器，避免被检测"""
    chrome_options = Options()
    
    # 简化配置，避免冲突（使用第一版的简单配置）
    chrome_options.add_argument('--start-maximized')
    chrome_options.add_argument('--disable-blink-features=AutomationControlled')
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    '''

    """配置Edge浏览器"""
    edge_options = Options()
    edge_options.add_argument('--start-maximized')
    edge_options.add_argument('--disable-blink-features=AutomationControlled')
    edge_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    edge_options.add_experimental_option('useAutomationExtension', False)
    
    try:
        
        # service = Service(ChromeDriverManager().install())
        # driver = webdriver.Chrome(service=service, options=chrome_options)

        service = Service(EdgeChromiumDriverManager().install())
        driver = webdriver.Edge(service=service, options=edge_options)
        print("✓ 浏览器启动成功")
        return driver
    except Exception as e:
        print(f"启动浏览器失败: {str(e)}")
        print("\n尝试使用更简单的配置...")

        
        # 如果失败，使用最简配置
        chrome_options = Options()
        chrome_options.add_argument('--start-maximized')
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        return driver


def scrape_iwf_results(url, output_file="weightlifting_results.xlsx", wait_time=15):
    """
    爬取IWF举重世锦赛数据
    """
    print("="*60)
    print("正在启动浏览器...")
    print("="*60)
    
    driver = setup_driver()
    
    try:
        print(f"\n正在访问: {url}")
        driver.get(url)
        
        # 等待页面加载
        print("等待页面加载...")
        time.sleep(5)
        
        # 给用户操作时间
        print("\n" + "="*60)
        print("【重要】浏览器已打开！")
        print("="*60)
        print(f"\n请在 {wait_time} 秒内完成以下操作：")
        print("1. 确认页面已正常加载（不是空白）")
        print("2. 点击【Men】或【Women】标签")
        print("3. 点击具体的重量级别（如 55kg, 61kg 等）")
        print("4. 等待运动员卡片数据加载完成")
        print(f"\n{wait_time}秒后将自动开始爬取...")
        print("="*60)
        
        for i in range(wait_time, 0, -1):
            print(f"\r倒计时: {i} 秒", end="", flush=True)
            time.sleep(1)
        print("\n\n开始爬取数据...")
        
        # 滚动页面确保所有数据加载
        print("正在滚动页面...")
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)
        driver.execute_script("window.scrollTo(0, 0);")
        time.sleep(1)
        
        # 获取页面源代码
        page_source = driver.page_source
        
        # 保存页面源代码
        with open("page_source.html", "w", encoding="utf-8") as f:
            f.write(page_source)
        print("✓ 页面源代码已保存到 page_source.html")
        
        # 解析HTML
        soup = BeautifulSoup(page_source, 'html.parser')
        
        # 查找cards容器
        cards_div = soup.find('div', class_='cards')
        
        if not cards_div:
            print("❌ 未找到cards容器！")
            print("请检查：")
            print("1. 页面是否正常加载（不是空白）")
            print("2. 是否已选择了具体的重量级别")
            print("3. 查看 page_source.html 确认页面内容")
            return None
        
        print("✓ 找到cards容器")
        
        # 查找所有card（排除第一个legend card）
        all_cards = cards_div.find_all('div', class_='card')
        
        # 过滤掉legend card（标题行）
        data_cards = [card for card in all_cards if 'card__legend' not in card.get('class', [])]
        
        print(f"✓ 找到 {len(data_cards)} 个运动员卡片")
        
        if len(data_cards) == 0:
            print("❌ 未找到运动员数据")
            return None
        
        # 解析每个card
        data = []
        
        for idx, card in enumerate(data_cards):
            try:
                row_data = {}
                
                # 提取所有<p>标签的文本
                all_p_tags = card.find_all('p')
                
                # 方法1: 按照HTML结构顺序提取
                # col-md-5: Rank, Name, Nation
                # col-md-4: Born, B.weight, Group
                # col-md-3: 1, 2, 3, Total
                
                # 提取Rank（第一个p标签）
                rank_p = card.find('div', class_='col-2').find('p') if card.find('div', class_='col-2') else None
                if rank_p:
                    rank_text = rank_p.get_text(strip=True)
                    # 移除 "Rank: " 前缀
                    row_data['Rank'] = rank_text.replace('Rank:', '').strip()
                
                # 提取Name（col-7）
                name_div = card.find('div', class_='col-7')
                if name_div:
                    name_p = name_div.find('p')
                    if name_p:
                        row_data['Name'] = name_p.get_text(strip=True)
                
                # 提取Nation（col-3，第一个）
                nation_divs = card.find_all('div', class_='col-3')
                if len(nation_divs) > 0:
                    nation_p = nation_divs[0].find('p')
                    if nation_p:
                        # 提取国家代码（去掉图片）
                        nation_text = nation_p.get_text(strip=True)
                        row_data['Nation'] = nation_text
                
                # 提取Born（col-5）
                born_div = card.find('div', class_='col-5')
                if born_div:
                    born_p = born_div.find('p', class_='normal__text')
                    if born_p:
                        born_text = born_p.get_text(strip=True)
                        row_data['Born'] = born_text.replace('Born:', '').strip()
                
                # 提取B.weight（col-4）
                weight_div = card.find('div', class_='col-4')
                if weight_div:
                    weight_p = weight_div.find('p', class_='normal__text')
                    if weight_p:
                        weight_text = weight_p.get_text(strip=True)
                        row_data['B.weight'] = weight_text.replace('B.weight:', '').strip()
                
                # 提取Group（col-3，第二个，在col-md-4内）
                col_md4 = card.find('div', class_='col-md-4')
                if col_md4:
                    group_div = col_md4.find('div', class_='col-3')
                    if group_div:
                        group_p = group_div.find('p', class_='normal__text')
                        if group_p:
                            group_text = group_p.get_text(strip=True)
                            row_data['Group'] = group_text.replace('Group:', '').strip()
                
                # 提取成绩（col-md-3内的4个col-3）
                col_md3 = card.find('div', class_='col-md-3')
                if col_md3:
                    result_divs = col_md3.find_all('div', class_='col-3')
                    
                    if len(result_divs) >= 4:
                        # 1st attempt (Snatch 1)
                        attempt1_p = result_divs[0].find('p', class_='normal__text')
                        if attempt1_p:
                            strong = attempt1_p.find('strong')
                            if strong:
                                text = strong.get_text(strip=True)
                                row_data['Snatch_1'] = text
                        
                        # 2nd attempt (Snatch 2)
                        attempt2_p = result_divs[1].find('p', class_='normal__text')
                        if attempt2_p:
                            strong = attempt2_p.find('strong')
                            if strong:
                                text = strong.get_text(strip=True)
                                row_data['Snatch_2'] = text
                        
                        # 3rd attempt (Snatch 3)
                        attempt3_p = result_divs[2].find('p', class_='normal__text')
                        if attempt3_p:
                            strong = attempt3_p.find('strong')
                            if strong:
                                text = strong.get_text(strip=True)
                                row_data['Snatch_3'] = text
                        
                        # Total
                        total_p = result_divs[3].find('p', class_='normal__text')
                        if total_p:
                            strong = total_p.find('strong')
                            if strong:
                                text = strong.get_text(strip=True)
                                row_data['Snatch_Total'] = text.replace('Total:', '').strip()
                
                # 只添加有效数据
                if row_data and row_data.get('Name'):
                    data.append(row_data)
                    if idx < 3:  # 显示前3条数据用于调试
                        print(f"\n运动员 {idx+1}: {row_data}")
                
            except Exception as e:
                print(f"解析第 {idx+1} 个card时出错: {str(e)}")
                continue
        
        if not data:
            print("❌ 未能解析出任何数据")
            print("请检查 page_source.html 确认页面结构")
            return None
        
        # 创建DataFrame
        df = pd.DataFrame(data)
        
        # 调整列顺序
        desired_columns = ['Rank', 'Name', 'Nation', 'Born', 'B.weight', 'Group', 
                          'Snatch_1', 'Snatch_2', 'Snatch_3', 'Snatch_Total']
        existing_columns = [col for col in desired_columns if col in df.columns]
        df = df[existing_columns]
        
        # 显示预览
        print("\n" + "="*60)
        print("数据预览:")
        print("="*60)
        print(df.head(10))
        print(f"\n总共 {len(df)} 条记录")
        print(f"列名: {list(df.columns)}")
        
        # 保存到Excel
        df.to_excel(output_file, index=False, engine='openpyxl')
        print(f"\n✓ 数据已保存到: {output_file}")
        
        return df
        
    except Exception as e:
        print(f"\n❌ 发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
        
    finally:
        print("\n按回车键关闭浏览器...")
        input()
        driver.quit()
        print("浏览器已关闭")


# 主程序
if __name__ == "__main__":
    print("="*60)
    print("IWF举重世锦赛数据爬虫")
    print("="*60)
    
    default_url = "https://iwf.sport/results/results-by-events/results-by-events-2018-2025/?event_id=577"
    
    print("\n使用说明：")
    print("1. 确保Clash VPN已开启")
    print("2. 浏览器会自动打开IWF网站")
    print("3. 手动点击 Men/Women 和重量级别")
    print("4. 等待倒计时结束，自动爬取\n")
    
    url = input(f"输入URL (直接回车使用默认):\n> ").strip() or default_url
    
    output_file = input("\n输入输出文件名 (如: men_55kg.xlsx): ").strip()
    if not output_file:
        output_file = "iwf_results.xlsx"
    if not output_file.endswith('.xlsx'):
        output_file += '.xlsx'
    
    wait_input = input("\n等待时间(秒，默认15秒): ").strip()
    wait_time = int(wait_input) if wait_input.isdigit() else 15
    
    print(f"\n开始爬取...")
    df = scrape_iwf_results(url, output_file, wait_time)
    
    if df is not None:
        print("\n" + "="*60)
        print("✓ 爬取成功！")
        print("="*60)
        print("\n提示：")
        print("- 这只是抓举(Snatch)的数据")
        print("- 如需挺举(C&J)数据，需要在网页上切换标签")
        print("- 建议分别保存不同重量级和项目的数据")
    else:
        print("\n" + "="*60)
        print("✗ 爬取失败")
        print("="*60)