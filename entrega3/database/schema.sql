-- Esquema generado automáticamente por database/crear_bd.py a partir de
-- backend/app/models.py. No editar a mano.

CREATE TABLE producto (
	id SERIAL NOT NULL, 
	clase_id INTEGER NOT NULL, 
	categoria_yolo VARCHAR(30) NOT NULL, 
	nombre VARCHAR(80) NOT NULL, 
	precio NUMERIC(10, 2) NOT NULL, 
	activo BOOLEAN NOT NULL, 
	PRIMARY KEY (id), 
	CONSTRAINT ck_producto_precio CHECK (precio >= 0), 
	UNIQUE (clase_id), 
	UNIQUE (categoria_yolo)
);

CREATE TABLE sesion_captura (
	id SERIAL NOT NULL, 
	codigo VARCHAR(10) NOT NULL, 
	descripcion VARCHAR(200) NOT NULL, 
	iluminacion VARCHAR(30) NOT NULL, 
	superficie VARCHAR(50) NOT NULL, 
	prueba_generalizacion BOOLEAN NOT NULL, 
	creado_en TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
	PRIMARY KEY (id), 
	UNIQUE (codigo)
);

CREATE TABLE archivo_media (
	id SERIAL NOT NULL, 
	sesion_id INTEGER NOT NULL, 
	tipo VARCHAR(10) NOT NULL, 
	nombre_original VARCHAR(200) NOT NULL, 
	ruta VARCHAR(300) NOT NULL, 
	tamano_bytes BIGINT NOT NULL, 
	estado VARCHAR(12) NOT NULL, 
	detalle VARCHAR(300), 
	creado_en TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
	PRIMARY KEY (id), 
	CONSTRAINT ck_archivo_tipo CHECK (tipo IN ('video', 'foto')), 
	CONSTRAINT ck_archivo_estado CHECK (estado IN ('subido', 'procesando', 'procesado', 'error')), 
	FOREIGN KEY(sesion_id) REFERENCES sesion_captura (id) ON DELETE CASCADE, 
	UNIQUE (ruta)
);

CREATE TABLE inventario (
	producto_id INTEGER NOT NULL, 
	cantidad INTEGER NOT NULL, 
	stock_minimo INTEGER NOT NULL, 
	actualizado_en TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
	PRIMARY KEY (producto_id), 
	CONSTRAINT ck_inventario_cantidad CHECK (cantidad >= 0), 
	CONSTRAINT ck_inventario_stock_minimo CHECK (stock_minimo >= 0), 
	FOREIGN KEY(producto_id) REFERENCES producto (id)
);

CREATE TABLE movimiento (
	id SERIAL NOT NULL, 
	producto_id INTEGER NOT NULL, 
	tipo VARCHAR(15) NOT NULL, 
	cantidad INTEGER NOT NULL, 
	precio_unitario NUMERIC(10, 2) NOT NULL, 
	fecha_hora TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
	origen VARCHAR(10) NOT NULL, 
	confianza FLOAT, 
	evidencia VARCHAR(300), 
	PRIMARY KEY (id), 
	CONSTRAINT ck_movimiento_tipo CHECK (tipo IN ('venta', 'reposicion', 'ajuste')), 
	CONSTRAINT ck_movimiento_origen CHECK (origen IN ('vision', 'manual')), 
	CONSTRAINT ck_movimiento_cantidad CHECK (cantidad <> 0), 
	FOREIGN KEY(producto_id) REFERENCES producto (id)
);

CREATE INDEX ix_movimiento_fecha ON movimiento (fecha_hora);

CREATE TABLE imagen (
	id SERIAL NOT NULL, 
	sesion_id INTEGER NOT NULL, 
	archivo_id INTEGER NOT NULL, 
	ruta VARCHAR(300) NOT NULL, 
	ancho INTEGER NOT NULL, 
	alto INTEGER NOT NULL, 
	estado VARCHAR(12) NOT NULL, 
	particion VARCHAR(5), 
	creado_en TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
	PRIMARY KEY (id), 
	CONSTRAINT ck_imagen_estado CHECK (estado IN ('pendiente', 'etiquetada', 'descartada')), 
	CONSTRAINT ck_imagen_particion CHECK (particion IS NULL OR particion IN ('train', 'val', 'test')), 
	FOREIGN KEY(sesion_id) REFERENCES sesion_captura (id) ON DELETE CASCADE, 
	FOREIGN KEY(archivo_id) REFERENCES archivo_media (id) ON DELETE CASCADE, 
	UNIQUE (ruta)
);

CREATE INDEX ix_imagen_sesion_estado ON imagen (sesion_id, estado);

CREATE TABLE anotacion (
	id SERIAL NOT NULL, 
	imagen_id INTEGER NOT NULL, 
	producto_id INTEGER NOT NULL, 
	x_centro DOUBLE PRECISION NOT NULL, 
	y_centro DOUBLE PRECISION NOT NULL, 
	ancho DOUBLE PRECISION NOT NULL, 
	alto DOUBLE PRECISION NOT NULL, 
	PRIMARY KEY (id), 
	CONSTRAINT ck_anotacion_centro CHECK (x_centro BETWEEN 0 AND 1 AND y_centro BETWEEN 0 AND 1), 
	CONSTRAINT ck_anotacion_tamano CHECK (ancho > 0 AND ancho <= 1 AND alto > 0 AND alto <= 1), 
	FOREIGN KEY(imagen_id) REFERENCES imagen (id) ON DELETE CASCADE, 
	FOREIGN KEY(producto_id) REFERENCES producto (id)
);

CREATE INDEX ix_anotacion_imagen_id ON anotacion (imagen_id);
