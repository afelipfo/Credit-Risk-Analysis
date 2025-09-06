import os
from pathlib import Path
import os

REDIS_IP = os.getenv("REDIS_IP", "redis")
REDIS_PORT = 6379
REDIS_DB_ID = 0
REDIS_QUEUE = "service_queue"
SERVER_SLEEP = 0.05

CITIES = str(Path(__file__).parent.parent / "src/auxiliar_files/Lista_Municípios_com_IBGE_Brasil_Versao_CSV.csv")