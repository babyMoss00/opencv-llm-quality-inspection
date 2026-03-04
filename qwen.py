"""
LLM-Powered Industrial Defect Diagnosis Pipeline
=================================================

Uses Qwen-VL-Max (multimodal LLM) for:
1. Few-shot defect recognition via reference image comparison
2. Structured JSON output with confidence scoring
3. Automatic HTML report generation

Usage:
    python qwen.py --target test_image.jpg --ref1 reference1.jpg --ref2 reference2.jpg
"""

import os
import argparse
import dashscope
import json
from dashscope import MultiModalConversation
from dotenv import load_dotenv

# 1. 环境配置
load_dotenv()
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")


# 2. 定义分析函数
def industrial_expert_analysis(image_path, ref_image1, ref_image2):
    """
    Multi-image context analysis using Qwen-VL-Max.

    Args:
        image_path: Path to the target image to inspect
        ref_image1: Path to reference image 1 (e.g., plastic film defect)
        ref_image2: Path to reference image 2 (e.g., membrane fiber defect)

    Returns:
        JSON string with defect analysis, or None on failure
    """
    if not dashscope.api_key:
        print("❌ 错误：未找到 API Key！请在 .env 文件中配置 DASHSCOPE_API_KEY")
        return None

    # 获取绝对路径并加上 file:// 协议头
    ref_img1 = f'file://{os.path.abspath(ref_image1)}'
    ref_img2 = f'file://{os.path.abspath(ref_image2)}'
    target_img = f'file://{os.path.abspath(image_path)}'

    # 结构化 Prompt：约束 LLM 输出格式，确保可解析
    prompt_content = """你是一名拥有 15 年经验的工业质检专家。
    请根据前两张图的特征，分析最后一张新图片。
    判断属于哪种异物，并严格按 JSON 格式输出，必须包含以下字段：
    - area: 区域位置
    - type: 缺陷细分类型
    - visual_evidence: 视觉特征描述
    - impact_analysis: 生产影响
    - professional_action: 具体修复动作建议
    - confidence: 置信度（0-1之间）

    输出格式示例：{"defects": [{"area": "左上", "type": "塑料薄片", ...}]}"""

    messages = [{
        'role': 'user',
        'content': [
            {'text': '这是标准异物对比。图 1 蓝色半透明且边缘锐利的是塑料薄片；图 2 蓝色且呈丝状的是膜丝。'},
            {'image': ref_img1},
            {'image': ref_img2},
            {'text': prompt_content},
            {'image': target_img},
        ]
    }]

    try:
        response = MultiModalConversation.call(model='qwen-vl-max', messages=messages)
        if response.status_code == 200:
            return response.output.choices[0].message.content[0]['text']
        else:
            print(f"API 调用失败：{response.message}")
            return None
    except Exception as e:
        print(f"请求发生异常：{e}")
        return None


# 3. 置信度重审逻辑 (Self-Correction)
def self_correction_review(image_path, ref_image1, ref_image2, initial_result):
    """
    When any defect confidence < 0.8, trigger re-review with stricter prompting.

    This reduces hallucination in edge cases by asking the model to:
    - Switch to a more rigorous analysis perspective
    - Exclude lighting/shadow interference
    - Re-evaluate with higher scrutiny
    """
    try:
        data = json.loads(initial_result.replace('```json', '').replace('```', '').strip())
        low_conf_defects = [
            d for d in data.get('defects', [])
            if d.get('confidence', 1.0) < 0.8
        ]

        if not low_conf_defects:
            return initial_result  # All confidence scores are high, no re-review needed

        print(f"⚠️ 检测到 {len(low_conf_defects)} 个低置信度缺陷，启动专家重审...")

        ref_img1 = f'file://{os.path.abspath(ref_image1)}'
        ref_img2 = f'file://{os.path.abspath(ref_image2)}'
        target_img = f'file://{os.path.abspath(image_path)}'

        review_prompt = f"""你是高级质检复审专家。初次检测发现以下低置信度结果：
{json.dumps(low_conf_defects, ensure_ascii=False, indent=2)}

请以更严苛的标准重新审视目标图片：
1. 排除光影干扰因素
2. 与参考图进行更细致的特征比对
3. 如确认为误报，将 confidence 设为 0 并说明原因
4. 如确认为真实缺陷，提供更详细的证据

严格按 JSON 格式输出：{{"defects": [...]}}"""

        messages = [{
            'role': 'user',
            'content': [
                {'text': '参考图 1（塑料薄片）和参考图 2（膜丝）：'},
                {'image': ref_img1},
                {'image': ref_img2},
                {'text': review_prompt},
                {'image': target_img},
            ]
        }]

        response = MultiModalConversation.call(model='qwen-vl-max', messages=messages)
        if response.status_code == 200:
            print("✅ 重审完成")
            return response.output.choices[0].message.content[0]['text']
        else:
            print(f"重审 API 调用失败，使用原始结果")
            return initial_result

    except Exception as e:
        print(f"重审解析失败: {e}，使用原始结果")
        return initial_result


# 4. 报告生成函数
def generate_html_report(ai_json_data, image_path):
    """Generate a styled HTML inspection report from LLM JSON output."""
    try:
        # 清洗 Markdown 标签
        clean_json = ai_json_data.replace('```json', '').replace('```', '').strip()
        data = json.loads(clean_json)

        items_html = ""
        for defect in data.get('defects', []):
            # 根据置信度设置颜色
            conf = defect.get('confidence', 0)
            color = "#ffcccc" if conf < 0.8 else "#e6fffa"

            items_html += f"""
            <div style="background-color: {color}; padding: 20px; margin: 15px 0; border-radius: 12px; border: 1px solid #ccc; line-height: 1.6;">
                <p style="font-size: 1.1em;"><strong>📍 区域位置：</strong> {defect.get('area', '未知')}</p>
                <p><strong>🔍 缺陷类型：</strong> {defect.get('type', '未知')}</p>
                <p><strong>👁️ 视觉特征：</strong> {defect.get('visual_evidence', '无描述')}</p>
                <p><strong>⚠️ 生产影响：</strong> {defect.get('impact_analysis', '未知')}</p>
                <p><strong>🛠️ 处置建议：</strong> <span style="color: #d9534f; font-weight: bold;">{defect.get('professional_action', '无')}</span></p>
                <p style="font-size: 0.9em; color: #666;">置信度评分：{conf * 100}%</p>
            </div>
            """

        html_template = f"""
        <html>
        <head><meta charset="utf-8"><title>高级 AI 工业质检报告</title></head>
        <body style="font-family: 'Microsoft YaHei', sans-serif; max-width: 900px; margin: 20px auto; padding: 20px; background-color: #f0f2f5;">
            <div style="background: white; padding: 30px; border-radius: 15px; box-shadow: 0 4px 20px rgba(0,0,0,0.1);">
                <h1 style="color: #333; text-align: center;">键盘模具 AI 深度质检报告</h1>
                <hr style="border: 0.5px solid #eee;">
                <div style="text-align: center; margin: 20px 0;">
                    <h3>检测目标图</h3>
                    <img src="file:///{os.path.abspath(image_path)}" style="max-width: 100%; border-radius: 10px; border: 2px solid #ddd;">
                </div>
                <h3>🔬 详细诊断报告：</h3>
                {items_html}
                <footer style="margin-top: 30px; text-align: center; color: #999; font-size: 0.8em;">
                    基于 Qwen-VL-Max 多图上下文推理生成
                </footer>
            </div>
        </body>
        </html>
        """
        with open("report.html", "w", encoding="utf-8") as f:
            f.write(html_template)
        print("✅ 深度报告已生成：请查看 report.html")
    except Exception as e:
        print(f"解析失败: {e}\nAI 返回原始数据: {ai_json_data}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM-Powered Industrial Defect Inspection")
    parser.add_argument("--target", default="target.jpg", help="Target image to inspect")
    parser.add_argument("--ref1", default="ref_plastic_film.jpg", help="Reference image 1 (plastic film)")
    parser.add_argument("--ref2", default="ref_membrane_fiber.jpg", help="Reference image 2 (membrane fiber)")
    parser.add_argument("--no-review", action="store_true", help="Skip self-correction review")
    args = parser.parse_args()

    print("🚀 正在启动专家级多模态分析...")
    raw_result = industrial_expert_analysis(args.target, args.ref1, args.ref2)

    if raw_result:
        # Self-Correction: 低置信度自动重审
        if not args.no_review:
            raw_result = self_correction_review(args.target, args.ref1, args.ref2, raw_result)

        generate_html_report(raw_result, args.target)
