
# --- Tower API endpoints ---
# Add these to server.py or as a separate router

from fastapi import APIRouter
tower_router = APIRouter(prefix="/tower", tags=["towers"])

@tower_router.get("/list")
def list_towers():
    """List all towers."""
    try:
        from hyphae.towers import get_all_towers
        towers = get_all_towers()
        return {"towers": towers}
    except Exception as e:
        return {"error": str(e)}

@tower_router.get("/walk/{name}")
def walk_tower_endpoint(name: str):
    """Walk through a tower — returns all floors, connections, views."""
    try:
        from hyphae.towers import walk_tower
        result = walk_tower(name)
        if result:
            return result
        return {"error": f"Tower '{name}' not found"}
    except Exception as e:
        return {"error": str(e)}

@tower_router.post("/seed")
def seed_tower_endpoint(name: str, purpose: str):
    """Manually seed a new tower."""
    try:
        from hyphae.towers import seed_tower
        tower_id = seed_tower(name, purpose)
        return {"tower_id": tower_id, "name": name, "status": "seeded"}
    except Exception as e:
        return {"error": str(e)}

@tower_router.post("/synthesize/{tower_id}")
def synthesize_endpoint(tower_id: str):
    """Trigger floor synthesis from accumulated materials."""
    try:
        from hyphae.towers import synthesize_floors, get_tower_floors
        synthesize_floors(tower_id)
        floors = get_tower_floors(tower_id)
        return {"floors": len(floors), "status": "synthesized"}
    except Exception as e:
        return {"error": str(e)}
