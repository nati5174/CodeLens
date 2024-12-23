from app import app, db
from sqlalchemy import inspect

with app.app_context():
    db.create_all()
    
    # Inspect and get the table names
    inspector = inspect(db.engine)
    tables = inspector.get_table_names()
    print(f"Tables in the database: {tables}")
