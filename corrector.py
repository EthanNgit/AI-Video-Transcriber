import json
import os
import time

import requests


class Corrector:

    def __init__(self):
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.gemini_url = os.getenv("GEMINI_URL")

        if not self.gemini_url or not self.gemini_api_key:
            raise ValueError(
                f"Corrector failed to instantiate, base url or api key missing from environment.")

        pass

    def post_process_transcripts(self, segments):
        """Post processes transcript json to catch errors, hallucinations, etc"""

        texts = [seg["text"] for seg in segments]

        # TODO: extract prompt
        prompt = (
            f"""
            You are an expert editor for the Mandarin Chinese dub of SpongeBob SquarePants.
            Your task is to correct an ordered JSON transcript. You will only correct specific character names and key terms to their official spellings.

            RULES:
            1.  **Preserve Original Text:** Your primary rule is to preserve the original text exactly, except for the terms in the 'Official Terms' list.
            2.  **Correct Only from List:** Only change text that is a clear phonetic or visual misspelling of an official term.
            3.  **Maintain JSON Structure:** The output JSON must have the exact same structure as the input.
            4.  **No Other Changes:** Do not fix grammar, add punctuation, or change any other words.
            5.  **Be Precise:** If "谢老板" appears, change it to "蟹老板". If "你好" appears, leave it as "你好".
            7.  When a character name below appears in english in the transcript, replace it with the mandarin version: "Larry" -> "拉里", "Sandy" -> "珊迪"
            6.  **Ending Hallucinations**: You are allowed to remove hallucinations, you must be sure it is one first though,
                    Examples of whisper hallucinations:
                        * Youtube style outros, such as: "谢谢大家", "本期视频就先说到这里了,欢迎订阅我的频道哦!", "下次见", "今天就到此为止".
                        * Copyrights/Websites/Links, such as: "example.com", "copyright by ..."
                        * Interpretations of sound, such as: "【海绵宝宝与蟹黄堡王互动声】"
                    **Make sure that these hallucinations do not make sense in the context**, if they do not replace with "",
                    lines that appear in hallucinations could also just be apart of the transcript

            ---

            # **COMPLETE SPONGEBOB MANDARIN TERM DICTIONARY**

            ## **Main Characters**

            * SpongeBob → 海绵宝宝
            * Patrick → 派大星
            * Squidward → 章鱼哥
            * Plankton → 痞老板
            * Mr. Krabs → 蟹老板
            * Pearl Krabs → 珍珍
            * Sandy Cheeks → 珊迪·奇克斯
            * Mermaid Man → 美人鱼战士
            * Barnacle Boy → 海星少年
            * Gary the Snail → 小蜗
            * Mrs. Puff → 泡芙老师
            * Karen (Plankton's wife) → 凯伦
            * Flying Dutchman → 飞天魔鬼
            * King Neptune → 海神
            * Larry the Lobster → 拉里
            * Bubble Bass → 海霸王
            * Anchovy → 凤尾鱼

            ## **Minor / Recurring Characters**

            * Patrick's Sister → 派大珊
            * Squilliam Fancyson → 章鱼威廉·弗克森
            * Patchy the Pirate → 海盗派奇 *(派斯 is uncommon; 正确为派奇)*
            * Rayman (Man Ray) → 射线恶魔
            * Dirty Bubble → 邪恶泡泡

            ---

            # **Locations (居所)**

            ### Homes

            * the Pineapple House → 菠萝屋 / 凤梨屋
            * Squidward's Moai House → 复活岛人像屋
            * Patrick's Rock House → 石头屋
            * Sandy's Treedome → 圆顶树屋
            * Mr. Krabs' Anchor House → 船锚屋
            * The Chum Bucket → 海之霸 *(痞老板餐厅)*

            ### General Setting

            * Bikini Bottom → 比基尼海滩 / 比奇堡 / 裤头村 *(央配、台配均存在)*
            * The Krusty Krab → 蟹堡王
            * Jellyfish Fields → 水母田
            * Mrs. Puff's Boating School → 泡芙女士的驾驶学校 / 划浪学校 *(版本差异)*

            ### Extra Canon Locations

            * Empty Island (Karate Island) → 空手道圣岛 / 竖笛圣岛
            * New Kelp City → 纽开普市
            * Shell City → 贝壳城
            * Fine Dining Restaurant (Squilliam episodes) → 帆船餐厅

            ---

            # **General Concepts & Creatures**

            * Jellyfish → 水母
            * Clams → 贝壳

            ---

            # **Optional Additions (Useful for correction logic)**

            These appear often in subtitles and may be misheard:

            * 蟹黄堡 (Krabby Patty)
            * 秘密配方 / 神秘配方 (Secret Recipe)
            * 打泡泡 / 泡泡艺术
            * 相机鱼 (generic news reporter fish)
            * 医生鱼 / 警察鱼 / 店员鱼

            ---

            You will be given the full JSON Array of strings. Return only valid JSON. Use double-quoted strings and no trailing commas.

            {json.dumps(texts, ensure_ascii=False)}
           """
        )

        max_retries = 5
        retry_delay = 2  # seconds

        for attempt in range(max_retries):
            try:
                print(f"Gemini attempt {attempt + 1}/{max_retries}...")
                text = self._call_gemini(prompt)
                cleaned_text = self._clean_ai_response(text)

                post_json_arr = json.loads(cleaned_text)

                if len(post_json_arr) != len(segments):
                    raise ValueError(
                        f"LLM returned wrong number of items. Expected {len(segments)}, got {len(post_json_arr)}")

                result = []
                for seg, new in zip(segments, post_json_arr):
                    s = dict(seg)
                    s["text"] = new
                    result.append(s)

                return result

            except (requests.exceptions.RequestException, json.JSONDecodeError, KeyError, IndexError, ValueError) as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    print("Max retries reached. Failing.")
                    raise e

    def _call_gemini(self, prompt: str) -> str:
        url = self.gemini_url
        headers = {"Content-Type": "application/json"}
        data = {
            "contents": [{"parts": [{"text": prompt}]}]
        }

        r = requests.post(url, headers=headers, params={"key": self.gemini_api_key}, json=data)
        r.raise_for_status()  # Check for HTTP errors
        out = r.json()

        if "candidates" not in out or not out["candidates"]:
            raise ValueError(f"No candidates returned from Gemini: {out}")

        cand = out["candidates"][0]

        if "content" not in cand:
            raise ValueError(f"No content in candidate: {cand}")

        text_output = cand["content"]["parts"][0].get("text", "").strip()

        return text_output

    def _clean_ai_response(self, input: str) -> str:
        text_output = input

        # Gemini Issue
        if text_output.startswith("```"):
            # remove the leading ```json or ```
            first_newline = text_output.find("\n")
            text_output = text_output[first_newline + 1:]
            # remove trailing ```
            if text_output.endswith("```"):
                text_output = text_output[:-3]
        text_output = text_output.strip()

        return text_output
