import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

url = "http://host.docker.internal:11434"

key_words_path: str = "config/key_words.json"
items_to_find_path: str = "config/items_to_find.json"

template = """{
    'invoice_number': 'verbatim-string',
    'invoice_date': 'date-time',
    'issuer': 'verbatim-string',
    'customer': 'verbatim-string',
    'email': 'verbatim-string',
    'total_amount': 'number',
    'sub_total': 'number',
    'IVA': 'number',
    'IVA 21%': 'number',
    'IVA 10 %': 'number'
}"""

items_to_find: list[str] = [
    'invoice number',
    'invoice date',
    'issuer name',
    'issuer NIF',
    'issuer address',
    'issuer city',
    'issuer postal code',
    'customer name',
    'total amount',
    'subtotal',
    'IVA 21%',
    'IVA 10%',
    'IVA 4%'
]

key_words: dict[str: str] = {
    'factura': 'invoice',
    'fecha': 'date',
    'emisor': 'issuer',
    'cliente': 'customer',
    'correo': 'email',
    'empresa': 'company',
    'número': 'number',
    'para': 'to',
    'de': 'from',
    'fra': 'invoice',
    'base imponible': 'sub_total'
}

item_table_words: list[str] = [
    'producto',
    'concepto',
    'article',
    'artículo',
    'articulo',
    'descripción',
    'descripcion'
]

total_table_words: list[str] = [
    'b. imponible',
    'base imponible',
    'total imponible',
    'total imponible'
]

taxes_values: list[int] = [
    4,
    10,
    21
]

tax_check_words: list[str] = ["iva",
                              "i.v.a."]

mapping_items_holded: dict[str, str] = {
    "invoice_number": "Num factura",
    "invoice_date": "Fecha dd/mm/yyyy",
    "invoice_due_date": "Fecha de vencimiento dd/mm/yyyy",
    "description": "Descripción",
    "issuer_name": "Nombre del contacto",
    "issuer_NIF": "NIF",
    "issuer_address": "Dirección",
    "issuer_city": "Población",
    "issuer_postal_code": "Código postal",
    "province": "Provincia",
    "country": "País",
    "item": "Concepto",
    "item_description": "Descripción del producto",
    "SKU": "SKU",
    "unit_price": "Precio unidad",
    "units": "Unidades",
    "discount": "Descuento %",
    "IVA": "IVA %",
    "retention": "Retención %",
    "Inv. Suj. Pasivo (1/0)": "Inv. Suj. Pasivo (1/0)",
    "operation": "Operación",
    "invoice_total": "Cantidad cobrada",
    "collection date": "Fecha de cobro",
    "payment account": "Cuenta de pago",
    "tags": "Tags separados por -",
    "payment_account_name": "Nombre cuenta de gasto",
    "payment_account_number": "Num. Cuenta de gasto",
    "currency": "Moneda",
    "exchange_rate": "Cambio de moneda",
    "warehouse": "Almacén"
}