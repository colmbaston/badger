#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 0) uniform UniformData
{
    mat4 projection_view;
    vec4 light_direction;
    vec4 camera_position;
}
uniform_data;

layout(location = 0) in  vec3 position;
layout(location = 1) in  vec3 normal;
layout(location = 0) out vec3 out_color;

void main()
{
    vec3 normal          = normalize(normal);
    vec3 light_direction = vec3(uniform_data.light_direction);
    vec3 camera_position = vec3(uniform_data.camera_position);

    float ambient  = 0.1;
    float diffuse  = dot(normal, light_direction);
    float specular = 0.0;

    if (diffuse > 0.0)
    {
        vec3 view_direction = normalize(camera_position - position);
        vec3 half_direction = normalize(light_direction + view_direction);
        specular            = pow(max(dot(normal, half_direction), 0.0), 16) * 0.5;
    }

    out_color = (max(ambient, diffuse) + specular) * vec3(1.0, 0.5, 0.0);
}
