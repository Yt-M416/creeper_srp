import pandas as pd
from selenium import webdriver
from selenium.webdriver.edge.service import Service
from selenium.webdriver.edge.options import Options
from webdriver_manager.microsoft import EdgeChromiumDriverManager
from bs4 import BeautifulSoup
import time

def setup_driver():
    """配置Edge浏览器"""
    edge_options = Options()
    edge_options.add_argument('--start-maximized')
    edge_options.add_argument('--disable-blink-features=AutomationControlled')
    edge_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    edge_options.add_experimental_option('useAutomationExtension', False)

    try:
        # ✅ 自动下载 EdgeDriver（从微软服务器，国内可访问）
        service = Service(EdgeChromiumDriverManager().install())
        driver = webdriver.Edge(service=service, options=edge_options)
        print("✓ Edge 浏览器启动成功")
        return driver
    except Exception as e:
        print(f"启动 Edge 浏览器失败: {str(e)}")
        raise

# 测试启动
if __name__ == "__main__":
    print("="*60)
    print("Edge 浏览器测试启动")
    print("="*60)
    driver = setup_driver()
    driver.get("https://iwf.sport/results/results-by-events/results-by-events-2018-2025/?event_id=577")
    time.sleep(5)
    driver.quit()
    print("✓ Edge 测试完成")
