import modal

app = modal.App("nwm-wandb-entity-diag")

image = modal.Image.debian_slim(python_version="3.11").pip_install("wandb")


@app.function(image=image, secrets=[modal.Secret.from_name("wandb-secret")], timeout=60)
def diag():
    import os
    import wandb

    api = wandb.Api(api_key=os.environ["WANDB_API_KEY"])
    v = api.viewer
    print("username:", v.username)
    print("entity:", v.entity)
    try:
        print("teams:", [t.name for t in v.teams])
    except Exception as e:
        print("teams lookup failed:", e)

    import json
    import requests

    query = """
    query Viewer {
        viewer {
            id
            username
            entity
            organizations {
                name
                orgType
                teams {
                    name
                }
            }
            teams {
                edges {
                    node {
                        name
                    }
                }
            }
        }
    }
    """
    try:
        r = requests.post(
            "https://api.wandb.ai/graphql",
            json={"query": query},
            auth=("api", os.environ["WANDB_API_KEY"]),
            timeout=30,
        )
        print(r.status_code)
        print(json.dumps(r.json(), indent=2))
    except Exception as e:
        print("raw graphql query failed:", e)


@app.local_entrypoint()
def main():
    diag.remote()
