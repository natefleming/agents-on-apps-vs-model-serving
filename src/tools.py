"""Tool definitions for the LangGraph agent.

These tools are shared across both deployment paths. Each tool uses
LangChain's @tool decorator so LangGraph can bind them to the LLM.
"""

from langchain_core.tools import tool


# Simulated product catalog
_PRODUCT_CATALOG: dict[str, dict[str, str | float | bool]] = {
    "laptop": {
        "name": "ProBook Laptop 15",
        "price": 1299.99,
        "in_stock": True,
        "category": "Electronics",
        "description": "15-inch display, 16GB RAM, 512GB SSD",
    },
    "headphones": {
        "name": "SoundMax Pro Headphones",
        "price": 249.99,
        "in_stock": True,
        "category": "Electronics",
        "description": "Noise-cancelling, wireless, 30hr battery",
    },
    "backpack": {
        "name": "TravelPro Backpack",
        "price": 89.99,
        "in_stock": False,
        "category": "Accessories",
        "description": "Water-resistant, laptop compartment, 35L",
    },
    "keyboard": {
        "name": "MechType Pro Keyboard",
        "price": 179.99,
        "in_stock": True,
        "category": "Electronics",
        "description": "Mechanical, RGB, wireless/wired, hot-swappable",
    },
    "monitor": {
        "name": "UltraView 27 Monitor",
        "price": 549.99,
        "in_stock": True,
        "category": "Electronics",
        "description": "27-inch 4K, USB-C, HDR400, adjustable stand",
    },
}


@tool
def lookup_product(product_name: str) -> str:
    """Look up product information by name.

    Use this tool when a customer asks about a specific product,
    its price, availability, or description.

    Args:
        product_name: The name or keyword of the product to look up.
                      Examples: 'laptop', 'headphones', 'keyboard'.
    """
    key = product_name.strip().lower()

    # Exact match
    if key in _PRODUCT_CATALOG:
        product = _PRODUCT_CATALOG[key]
        stock_status = "In Stock" if product["in_stock"] else "Out of Stock"
        return (
            f"Product: {product['name']}\n"
            f"Price: ${product['price']:.2f}\n"
            f"Status: {stock_status}\n"
            f"Category: {product['category']}\n"
            f"Description: {product['description']}"
        )

    # Fuzzy match: check if the search term appears in any product key or name
    matches: list[str] = []
    for prod_key, product in _PRODUCT_CATALOG.items():
        if key in prod_key or key in str(product["name"]).lower():
            matches.append(prod_key)

    if matches:
        results = []
        for match_key in matches:
            p = _PRODUCT_CATALOG[match_key]
            stock = "In Stock" if p["in_stock"] else "Out of Stock"
            results.append(f"- {p['name']}: ${p['price']:.2f} ({stock})")
        return f"Found {len(matches)} matching product(s):\n" + "\n".join(results)

    available = ", ".join(_PRODUCT_CATALOG.keys())
    return f"Product '{product_name}' not found. Available products: {available}"


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression and return the result.

    Use this tool for any calculations like totals, discounts,
    tax computations, or unit conversions.

    Args:
        expression: A mathematical expression to evaluate.
                    Examples: '1299.99 * 0.9', '249.99 + 179.99', '100 / 3'.
    """
    # Restrict to safe characters for math evaluation
    allowed_chars = set("0123456789+-*/.() ")
    if not all(c in allowed_chars for c in expression):
        return f"Error: Expression contains invalid characters. Only numbers and +-*/.() are allowed."

    try:
        result = eval(expression)  # noqa: S307 - restricted to numeric chars above
        if isinstance(result, float):
            return f"{expression} = {result:.4f}"
        return f"{expression} = {result}"
    except (SyntaxError, ZeroDivisionError, TypeError) as e:
        return f"Error evaluating '{expression}': {e}"


# Export all tools as a list for easy binding
ALL_TOOLS: list = [lookup_product, calculator]
