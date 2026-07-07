"""Sample MCP server for the whetkit demo: a small e-commerce backend.

This server is INTENTIONALLY badly curated — that is the point. It exposes
too many tools, with cryptic or misleading names, vague descriptions, and
duplicated functionality, so that `whetkit curate` has something to fix:

- ``data_query_1`` / ``legacy_search`` both search products; neither says so.
- ``data_query_2`` searches orders but is described like a generic query.
- ``get_rec`` / ``fetch_record`` are exact duplicates.
- ``do_thing`` sends an order confirmation; the name says nothing.
- ``list_all_products_with_optional_filters_and_pagination_v3`` works fine
  but the name wastes tokens.
- ``admin_reset``, ``sys_ping``, ``util_helper`` are noise for agent tasks.

The data is deterministic so eval tasks are reproducible. The tools only
mutate in-process state; running them has no side effects on your machine.
"""

import json
from copy import deepcopy

from mcp.server.fastmcp import FastMCP

server = FastMCP("shopdb")

_INITIAL_PRODUCTS = {
    "P-1": {"id": "P-1", "name": "AeroGlide Wireless Mouse", "price": 24.99, "stock": 132},
    "P-2": {"id": "P-2", "name": "ThunderClick Wireless Mouse Pro", "price": 49.99, "stock": 41},
    "P-3": {"id": "P-3", "name": "KeyForge Mechanical Keyboard", "price": 89.0, "stock": 17},
    "P-4": {"id": "P-4", "name": "GlassView 27in Monitor", "price": 219.0, "stock": 8},
    "P-5": {"id": "P-5", "name": "PixelPad Drawing Tablet", "price": 129.5, "stock": 0},
}

_INITIAL_CUSTOMERS = {
    "CUST-2": {"id": "CUST-2", "name": "Amina Alaoui", "email": "amina@example.com"},
    "CUST-7": {"id": "CUST-7", "name": "Jonas Weber", "email": "jonas@example.com"},
}

_INITIAL_ORDERS = {
    "ORD-1001": {
        "id": "ORD-1001",
        "customer_id": "CUST-2",
        "items": [{"product_id": "P-1", "quantity": 1}],
        "status": "delivered",
    },
    "ORD-1002": {
        "id": "ORD-1002",
        "customer_id": "CUST-7",
        "items": [{"product_id": "P-3", "quantity": 2}],
        "status": "shipped",
    },
}

PRODUCTS = deepcopy(_INITIAL_PRODUCTS)
CUSTOMERS = deepcopy(_INITIAL_CUSTOMERS)
ORDERS = deepcopy(_INITIAL_ORDERS)
SENT_NOTIFICATIONS: list[dict] = []
_order_counter = 1002


def _search_products(query: str) -> list[dict]:
    words = query.lower().split()
    return [p for p in PRODUCTS.values() if all(w in p["name"].lower() for w in words)]


@server.tool()
def data_query_1(q: str) -> str:
    """Query data."""
    return json.dumps(_search_products(q))


@server.tool()
def legacy_search(term: str) -> str:
    """Search (legacy)."""
    return json.dumps(_search_products(term))


@server.tool()
def data_query_2(q: str) -> str:
    """Query data from the system."""
    q_lower = q.lower()
    hits = [
        o
        for o in ORDERS.values()
        if q_lower in o["id"].lower()
        or q_lower in o["customer_id"].lower()
        or q_lower in o["status"]
    ]
    return json.dumps(hits)


@server.tool()
def list_all_products_with_optional_filters_and_pagination_v3(
    name_contains: str = "", max_price: float = 0.0, page: int = 1, page_size: int = 50
) -> str:
    """Lists all of the products that exist in the product catalog database table
    with optional filters for the name field (substring match) and the price field
    (maximum price, 0 means no maximum) as well as optional pagination controls for
    the page number and the page size which defaults to fifty items per page."""
    items = list(PRODUCTS.values())
    if name_contains:
        items = [p for p in items if name_contains.lower() in p["name"].lower()]
    if max_price > 0:
        items = [p for p in items if p["price"] <= max_price]
    start = (page - 1) * page_size
    return json.dumps(items[start : start + page_size])


@server.tool()
def get_rec(rec_id: str) -> str:
    """Get record."""
    rec = CUSTOMERS.get(rec_id)
    return json.dumps(rec) if rec else json.dumps({"error": f"no customer {rec_id}"})


@server.tool()
def fetch_record(record_identifier: str) -> str:
    """Fetches a record from the store."""
    rec = CUSTOMERS.get(record_identifier)
    return json.dumps(rec) if rec else json.dumps({"error": f"no customer {record_identifier}"})


@server.tool()
def cust_upd(cust_id: str, email: str) -> str:
    """Update cust."""
    customer = CUSTOMERS.get(cust_id)
    if not customer:
        return json.dumps({"error": f"no customer {cust_id}"})
    customer["email"] = email
    return json.dumps(customer)


@server.tool()
def proc_ord(customer_id: str, product_id: str, quantity: int) -> str:
    """Process."""
    global _order_counter
    if customer_id not in CUSTOMERS:
        return json.dumps({"error": f"no customer {customer_id}"})
    product = PRODUCTS.get(product_id)
    if not product:
        return json.dumps({"error": f"no product {product_id}"})
    if product["stock"] < quantity:
        return json.dumps({"error": f"insufficient stock for {product_id}"})
    _order_counter += 1
    order_id = f"ORD-{_order_counter}"
    ORDERS[order_id] = {
        "id": order_id,
        "customer_id": customer_id,
        "items": [{"product_id": product_id, "quantity": quantity}],
        "status": "pending",
    }
    product["stock"] -= quantity
    return json.dumps(ORDERS[order_id])


@server.tool()
def ord_status_check_tool_v2(order_ref: str) -> str:
    """Tool for checking. Version 2."""
    order = ORDERS.get(order_ref)
    return json.dumps(order) if order else json.dumps({"error": f"no order {order_ref}"})


@server.tool()
def inv_check(pid: str) -> str:
    """Inv."""
    product = PRODUCTS.get(pid)
    if not product:
        return json.dumps({"error": f"no product {pid}"})
    return json.dumps({"product_id": pid, "stock": product["stock"]})


@server.tool()
def do_thing(target: str, ref: str) -> str:
    """Does the thing for the given target and ref."""
    SENT_NOTIFICATIONS.append({"to": target, "order_id": ref})
    return json.dumps({"sent": True, "to": target, "order_id": ref})


@server.tool()
def util_helper(amount: float, currency: str = "USD") -> str:
    """Helper utility."""
    return f"{amount:,.2f} {currency}"


@server.tool()
def sys_ping() -> str:
    """Ping."""
    return "pong"


@server.tool()
def admin_reset(confirm: bool = False) -> str:
    """Admin use only."""
    if not confirm:
        return json.dumps({"error": "confirm required"})
    global PRODUCTS, CUSTOMERS, ORDERS, _order_counter
    PRODUCTS = deepcopy(_INITIAL_PRODUCTS)
    CUSTOMERS = deepcopy(_INITIAL_CUSTOMERS)
    ORDERS = deepcopy(_INITIAL_ORDERS)
    SENT_NOTIFICATIONS.clear()
    _order_counter = 1002
    return json.dumps({"reset": True})


if __name__ == "__main__":
    server.run()
