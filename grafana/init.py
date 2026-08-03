import json
import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

GRAFANA_URL = "http://grafana:3000"

USER = os.getenv("GRAFANA_ADMIN_USER", "admin")
PASSWORD = os.getenv("GRAFANA_ADMIN_PASSWORD", "admin")

PG_HOST = os.getenv("POSTGRES_HOST", "postgres")
PG_DB = os.getenv("POSTGRES_DB", "study_assistant")
PG_USER = os.getenv("POSTGRES_USER", "postgres")
PG_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")
PG_PORT = os.getenv("POSTGRES_PORT", "5432")

AUTH = (USER, PASSWORD)

def wait_for_grafana():

    while True:

        try:

            r = requests.get(
                f"{GRAFANA_URL}/api/health"
            )

            if r.status_code == 200:

                print("Grafana ready")
                return

        except Exception:

            pass

        print("Waiting for Grafana...")
        time.sleep(3)

def create_datasource():

    r = requests.get(
        f"{GRAFANA_URL}/api/datasources/name/PostgreSQL",
        auth=AUTH
    )

    if r.status_code == 200:

        uid = r.json()["uid"]

        print(
            "Datasource exists:",
            uid
        )

        return uid

    payload = {
        "name": "PostgreSQL",
        "type": "postgres",
        "access": "proxy",

        "url": f"{PG_HOST}:{PG_PORT}",

        "user": PG_USER,

        "jsonData": {
            "database": PG_DB,
            "sslmode": "disable",
            "postgresVersion": 1600
        },

        "secureJsonData": {
            "password": PG_PASSWORD
        },

        "isDefault": True
    }

    r = requests.post(

        f"{GRAFANA_URL}/api/datasources",

        auth=AUTH,

        json=payload

    )

    if r.status_code not in (200, 409):

        print(
            "Datasource creation failed:",
            r.text
        )

        raise Exception(
            "Datasource failed"
        )

    r = requests.get(

        f"{GRAFANA_URL}/api/datasources/name/PostgreSQL",

        auth=AUTH

    )

    uid = r.json()["uid"]

    print(
        "Datasource created:",
        uid
    )

    return uid


def create_dashboard(datasource_uid):

    BASE_DIR = os.path.dirname(
        os.path.abspath(__file__)
    )

    dashboard_file = os.path.join(
        BASE_DIR,
        "dashboard.json"
    )

    with open(dashboard_file) as f:

        dashboard = json.load(f)

    dashboard.pop(
        "uid",
        None
    )

    for panel in dashboard.get(
        "panels",
        []
    ):

        if "datasource" in panel:

            panel["datasource"]["uid"] = datasource_uid

        for target in panel.get(
            "targets",
            []
        ):

            if "datasource" in target:

                target["datasource"]["uid"] = datasource_uid

    payload = {

        "dashboard": dashboard,

        "overwrite": True,

        "folderId": 0

    }

    r = requests.post(

        f"{GRAFANA_URL}/api/dashboards/db",

        auth=AUTH,

        json=payload

    )

    if r.status_code == 200:

        print(
            "Dashboard imported:"
        )

        print(
            r.json()["url"]
        )

    else:

        print(
            "Dashboard import failed:"
        )

        print(
            r.text
        )

        raise Exception(
            "Dashboard failed"
        )

if __name__ == "__main__":

    wait_for_grafana()

    datasource_uid = create_datasource()

    create_dashboard(
        datasource_uid
    )

    print(
        "Grafana initialization complete"
    )