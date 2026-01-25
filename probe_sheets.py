# probe_sheets.py
import gspread, json, os, sys
from google.oauth2.service_account import Credentials
from config import GOOGLE_APPLICATION_CREDENTIALS, GSHEET_SPREADSHEET_ID, GSHEET_WORKSHEET_NAME
SCOPE=["https://www.googleapis.com/auth/spreadsheets.readonly"]
print("Creds:", GOOGLE_APPLICATION_CREDENTIALS)
print("Spreadsheet ID:", GSHEET_SPREADSHEET_ID)
print("Worksheet:", GSHEET_WORKSHEET_NAME)
creds=Credentials.from_service_account_file(GOOGLE_APPLICATION_CREDENTIALS, scopes=SCOPE)
gc=gspread.authorize(creds)
ws=gc.open_by_key(GSHEET_SPREADSHEET_ID).worksheet(GSHEET_WORKSHEET_NAME)
rows=ws.get_all_records()
print("OK. Rows loaded:", len(rows))
print("First row keys:", list(rows[0].keys()) if rows else [])
