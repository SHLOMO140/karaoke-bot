# פריסה חינמית 24/7 — Hugging Face Spaces (Gradio SDK, בלי כרטיס)

הבוט רץ ב-thread רקע (long-polling של טלגרם) לצד דף סטטוס של Gradio על פורט 7860.
וידאו מעל 50MB מוגש דרך מנגנון `/file=` המובנה של Gradio. הספרייה נשארת ב-Lovable/Supabase.
(הערה: סוג "Docker" של HF הפך בתשלום — לכן משתמשים ב-Gradio החינמי.)

## 1. יצירת Space

1. היכנס ל-https://huggingface.co (הרשמה חינם, **בלי כרטיס אשראי**).
2. **New Space** → SDK: **Gradio** → Blank → Private.
3. העלה את קבצי הבוט: לשונית **Files** → **Add file → Upload files** → גרור את התיקייה
   הנקייה (`karaoke-deploy`). כוללת `packages.txt` (ffmpeg+nodejs מותקנים אוטומטית).

## 2. Secrets (ב-Space → Settings → Variables and secrets)

הדבק את הערכים **כאן ב-Space**, לא בקוד ולא בצ'אט:

| שם | ערך |
|---|---|
| `BOT_TOKEN` | הטוקן מ-BotFather (או שמור `bot_token.txt` בריפו — הוא ב-gitignore) |
| `SUPABASE_URL` | כתובת ה-Supabase של פרויקט Lovable (Supabase → Settings → API → Project URL) |
| `SUPABASE_SERVICE_ROLE_KEY` | מפתח `service_role` (Supabase → Settings → API). מפתח-על — סוד! |
| `PUBLIC_BASE_URL` | כתובת ה-Space, למשל `https://<user>-<space>.hf.space` |
| `YTDLP_COOKIE_FILE` | `/app/cookies.txt` (ראה סעיף 3) |

## 3. Cookies של יוטיוב (חובה כדי לעקוף חסימת IP של הענן)

1. ייצא `cookies.txt` בפורמט Netscape מדפדפן מחובר ליוטיוב (עדיף חשבון משני).
2. העלה אותו כ-**secret file** ב-Space בשם `cookies.txt` (או הוסף לריפו — הוא ב-gitignore
   מקומית, אז ב-Space העלה אותו ידנית ל-`/app/cookies.txt`).
3. ודא ש-`YTDLP_COOKIE_FILE=/app/cookies.txt`.

## 4. פינג שמונע שינה (חינם, בלי כרטיס)

Space חינמי נכנס לשינה אחרי ~48 שעות בלי בקשות HTTP. כדי שירוץ 24/7:

1. היכנס ל-https://cron-job.org (חינם) או UptimeRobot.
2. צור job שמבצע GET ל-`https://<user>-<space>.hf.space/` כל 5 דקות.

## 5. בדיקה מקצה-לקצה

1. פתח את הבוט בטלגרם → `/start`.
2. שלח שם שיר → בחר תוצאה → **אקורדים** (בדוק שהשיר הופיע/עודכן ב-Supabase `songs`).
3. **הורדת השיר** → MP3 → אמור להגיע קובץ + "✅ הועלה בהצלחה".
4. **הורדת השיר** → וידאו → איכות. קובץ קטן מגיע ישירות; גדול מ-50MB מגיע כקישור.

## הרצה מקומית

```bash
pip install -r requirements.txt          # + ffmpeg + node ב-PATH
python app.py                            # קורא bot_token.txt / משתני סביבה
# בדיקה: GET http://localhost:7860/ מחזיר "ok"
```
