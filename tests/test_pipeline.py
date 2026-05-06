import pytest
import asyncio

from app.graph.pipeline import prepare_generation


def test_prepare_generation_validates_payload() -> None:
    async def run_test() -> None:
        with pytest.raises(ValueError, match="`messages` is required"):
            await prepare_generation({"model": "dnd-skill-router"})

    asyncio.run(run_test())
