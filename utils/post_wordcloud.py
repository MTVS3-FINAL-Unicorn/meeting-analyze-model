import httpx
from fastapi import HTTPException


async def post_wordcloud(file_path, key, meeting_id):   
    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            file = {
                "wordcloudFile": (key, open(file_path, "rb"))
            }
            data = {
                "meetingId": meeting_id
            }
            response = await client.post("http://125.132.216.190:319/api/v1/report/whole/wordcloud", files=file, data=data)
            response.raise_for_status()
            
            return response.text
        
        except httpx.HTTPStatusError as exc:
            raise HTTPException(status_code=exc.response.status_code, detail=str(exc))
