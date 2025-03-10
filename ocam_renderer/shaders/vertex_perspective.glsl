#version 330 core

layout(location = 0) in vec3 vertexPos;
layout(location = 1) in vec3 vertexColor;
layout(location = 2) in int vertexId;

out VS_OUT {
    vec3 color;
	int inst_id;
	float depth;
} vs_out;

uniform mat4 MV;
uniform mat4 P;

void main(){
	vec4 vertexPosMV = MV * vec4(vertexPos, 1);
	gl_Position = P * vertexPosMV;
	vs_out.color = vertexColor;
	vs_out.inst_id = vertexId;
	vs_out.depth = abs(vertexPosMV.z);
}