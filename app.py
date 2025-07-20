from flask import Flask, request, jsonify
import pandas as pd
import mysql.connector
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# Función para conectarse y construir el modelo de recomendaciones
def construir_modelo():
    conn = mysql.connector.connect(
        host="mx32.hostgator.mx",     
        user="marlenhe_Marlen",       
        password="Marlen1602",        
        database="marlenhe_smokeygrill"  
    )

    query = """
        SELECT p.usuarioId, dp.productoId, dp.cantidad
        FROM pedidos p
        JOIN detalle_pedido dp ON p.id = dp.pedidoId
    """

    df = pd.read_sql(query, conn)
    conn.close()

    if df.empty:
        return None, None

    # Crear tabla de usuario-producto
    pivot = df.pivot_table(index='usuarioId', columns='productoId', values='cantidad', aggfunc='sum', fill_value=0)

    # Escalar los valores
    scaler = MinMaxScaler()
    pivot_scaled = scaler.fit_transform(pivot)

    # Calcular similitud entre usuarios
    similarity_matrix = cosine_similarity(pivot_scaled)
    similarity_df = pd.DataFrame(similarity_matrix, index=pivot.index, columns=pivot.index)

    return pivot, similarity_df

# Construir modelo al iniciar el servidor
pivot, similarity_df = construir_modelo()

@app.route('/api/recomendar', methods=['GET'])
def recomendar():
    usuario_id = request.args.get('usuarioId', type=int)

    if pivot is None or similarity_df is None:
        return jsonify({"error": "No hay datos para generar recomendaciones"}), 500

    if usuario_id not in pivot.index:
        return jsonify([])  # Usuario sin historial

    similares = similarity_df[usuario_id].sort_values(ascending=False)[1:]
    productos_usuario = set(pivot.loc[usuario_id][pivot.loc[usuario_id] > 0].index)

    recomendaciones = {}
    for similar_usuario, score in similares.items():
        productos_similar = pivot.loc[similar_usuario]
        for producto_id, cantidad in productos_similar.items():
            if producto_id not in productos_usuario and cantidad > 0:
                if producto_id not in recomendaciones:
                    recomendaciones[producto_id] = 0
                recomendaciones[producto_id] += score * cantidad

    productos_ordenados = sorted(recomendaciones.items(), key=lambda x: x[1], reverse=True)
    recomendados = [int(producto_id) for producto_id, _ in productos_ordenados[:5]]

    return jsonify(recomendados)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
