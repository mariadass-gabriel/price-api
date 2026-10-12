from pydantic import BaseModel, ConfigDict, Field


class PredictRequest(BaseModel):
    """Features of one product, as scraped from a shop page."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "company_name": "pccomponentes",
                "brand_name": "Gigabyte",
                "category": "Discos Duros",
                "name": "Gigabyte SSD M.2 512GB 2280 PCIe x2 NVMe",
                "score": 9.5,
                "reviews": 40,
            }
        },
    )

    company_name: str = Field(min_length=1, max_length=100, description="Shop the product was scraped from")
    brand_name: str = Field(min_length=1, max_length=100)
    category: str = Field(min_length=1, max_length=100, description="Product category (Spanish shop labels)")
    name: str = Field(min_length=1, max_length=300, description="Product title, e.g. 'Kingston SSD 480GB SATA'")
    score: float = Field(ge=0, le=10, description="Average user rating, 0-10")
    reviews: int = Field(ge=0, le=1_000_000, description="Number of user reviews")


class PredictResponse(BaseModel):
    price: float = Field(description="Predicted price (EUR)")
    low: float = Field(description="Lower bound of the 90 % prediction interval")
    high: float = Field(description="Upper bound of the 90 % prediction interval")
    coverage: float = Field(description="Nominal coverage of the interval (1 - alpha)")


class HealthResponse(BaseModel):
    status: str
    model: str
