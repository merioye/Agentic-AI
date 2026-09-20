from langchain.tools import tool

# ---------------------------------------------------------------------------
# Billing tools
# ---------------------------------------------------------------------------

_MOCK_INVOICES = {
    "cust_001": {"invoice_id": "INV-9001", "amount": 49.00, "status": "paid", "date": "2026-08-01"},
    "cust_002": {"invoice_id": "INV-9002", "amount": 129.00, "status": "overdue", "date": "2026-07-15"}
}


@tool
def look_up_invoice(customer_id: str) -> str:
    """Retrieve the most recent invoice for a customer by their customer ID."""
    invoice = _MOCK_INVOICES.get(customer_id)
    if not invoice:
        return f"No invoice found for customer {customer_id}."
    return (
        f"Invoice {invoice['invoice_id']} for {customer_id}: "
        f"${invoice['amount']:.2f}, status={invoice['status']}, dated {invoice['data']}."
    )


@tool
def process_refund(invoice_id: str, amount: float) -> str:
    """
    Process a refund against a specific invoice ID for a given dollar amount.
    Only call this after confirming the invoice ID and amount are correct.
    """
    return f"Refund of ${amount:.2f} processed against invoice {invoice_id}. Funds return in 3-5 business days."


BILLING_TOOLS = [look_up_invoice, process_refund]


# ---------------------------------------------------------------------------
# Technical support tools
# ---------------------------------------------------------------------------

_KNOWN_ISSUES = {
    "login": "Known issue: password reset emails are delayed up to 10 minutes during peak hours.",
    "sync": "Known issue: mobile app sync can lag if the app was background for >24 hours; force-refresh fixes it."
}


@tool
def search_known_issues(keyword: str) -> str:
    """Search the known-issues knowledge base for a keyword (e.g. 'login', 'sync', 'crash')."""
    for key, description in _KNOWN_ISSUES.items():
        if key in keyword.lower():
            return description
    return "No matching known issue found. This may require escalation to engineering."


@tool
def create_engineering_ticket(summary: str, severity: str) -> str:
    """
    File a ticket with the engineering team for an issue that isn't a known,
    already-documented problem. Severity must be one of: low, medium, high, critical.
    """
    return f"Engineering ticket created (severity={severity}): '{summary}'. Reference: ENG-4471."


TECHNICAL_TOOLS = [search_known_issues, create_engineering_ticket]


# ---------------------------------------------------------------------------
# Order status tools
# ---------------------------------------------------------------------------

_MOCK_ORDERS = {
    "ord_100": {"status": "out for delivery", "eta": "tomorrow"},
    "ord_101": {"status": "processing", "eta": "3-5 business days"}
}


@tool
def get_order_status(order_id: str) -> str:
    """Look up the current shipping status and ETA of an order by its order ID."""
    order = _MOCK_ORDERS.get(order_id)
    if not order:
        return f"No order found with ID {order_id}"
    return f"Order {order_id}: {order['status']}, ETA {order['eta']}."


ORDER_TOOLS = [get_order_status]