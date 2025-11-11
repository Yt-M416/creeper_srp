import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

def check_font(font_name):
    """检查特定字体是否可用"""
    fonts = [f.name for f in fm.fontManager.ttflist]
    if font_name in fonts:
        print(f"✓ '{font_name}' 已安装")
        return True
    else:
        print(f"✗ '{font_name}' 未找到")
        return False

# 测试常见中文字体
test_fonts = ['SimHei', 'Microsoft YaHei', 'PingFang SC', 
              'WenQuanYi Zen Hei', 'SimSun']

print("检查中文字体:")
for font in test_fonts:
    check_font(font)