package com.juraj.agsl

import android.graphics.BitmapFactory
import android.graphics.RenderEffect
import android.graphics.RuntimeShader
import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.animation.core.withInfiniteAnimationFrameMillis
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.wrapContentSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.State
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.produceState
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asComposeRenderEffect
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.layout.onSizeChanged
import androidx.compose.ui.platform.LocalConfiguration
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.view.WindowCompat
import com.juraj.agsl.ui.theme.AGSLShadersTheme
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

private const val SNOW_SHADER_SRC = """
uniform float2 size;
uniform float time;
uniform shader composable;
float rotationAngle = 0.0;

float2 mod289(float2 x) {
    return x - floor(x * (1.0 / 289.0)) * 289.0;
}

float3 mod289(float3 x) {
    return x - floor(x * (1.0 / 289.0)) * 289.0;
}

float4 mod289(float4 x) {
    return x - floor(x * (1.0 / 289.0)) * 289.0;
}

float3 permute(float3 x) {
    return mod289(((x * 34.0) + 1.0) * x);
}

float4 permute(float4 x) {
    return mod((34.0 * x + 1.0) * x, 289.0);
}

float4 taylorInvSqrt(float4 r) {
    return 1.79284291400159 - 0.85373472095314 * r;
}

float snoise(float2 v) {
    const float4 C = float4(0.211324865405187, 0.366025403784439, -0.577350269189626, 0.024390243902439);
    float2 i = floor(v + dot(v, C.yy));
    float2 x0 = v - i + dot(i, C.xx);

    float2 i1 = x0.x > x0.y ? float2(1.0, 0.0) : float2(0.0, 1.0);
    float4 x12 = x0.xyxy + C.xxzz;
    x12.xy -= i1;

    i = mod289(i);
    float3 p = permute(permute(i.y + float3(0.0, i1.y, 1.0)) + i.x + float3(0.0, i1.x, 1.0));

    float3 m = max(0.5 - float3(dot(x0, x0), dot(x12.xy, x12.xy), dot(x12.zw, x12.zw)), 0.0);
    m = m * m;
    m = m * m;

    float3 x = 2.0 * fract(p * C.www) - 1.0;
    float3 h = abs(x) - 0.5;
    float3 ox = floor(x + 0.5);
    float3 a0 = x - ox;

    m *= 1.79284291400159 - 0.85373472095314 * (a0 * a0 + h * h);

    float3 g;
    g.x = a0.x * x0.x + h.x * x0.y;
    g.yz = a0.yz * x12.xz + h.yz * x12.yw;

    return 130.0 * dot(m, g);
}

float cellular2x2(float2 P) {
    const float K = 0.142857142857;
    const float K2 = 0.0714285714285;
    const float jitter = 0.8;

    float2 Pi = mod(floor(P), 289.0);
    float2 Pf = fract(P);
    float4 Pfx = Pf.x + float4(-0.5, -1.5, -0.5, -1.5);
    float4 Pfy = Pf.y + float4(-0.5, -0.5, -1.5, -1.5);
    float4 p = permute(Pi.x + float4(0.0, 1.0, 0.0, 1.0));
    p = permute(p + Pi.y + float4(0.0, 0.0, 1.0, 1.0));
    float4 ox = mod(p, 7.0) * K + K2;
    float4 oy = mod(floor(p * K), 7.0) * K + K2;
    float4 dx = Pfx + jitter * ox;
    float4 dy = Pfy + jitter * oy;
    float4 d = dx * dx + dy * dy;

    d.xy = min(d.xy, d.zw);
    d.x = min(d.x, d.y);
    return d.x;
}

float fbm(float2 p) {
    float f = 0.0;
    float w = 0.5;
    for (int i = 0; i < 5; i++) {
        f += w * snoise(p);
        p *= 2.0;
        w *= 0.5;
    }
    return f;
}

float2 rotate(float2 uv, float angle) {
    float cosAngle = cos(angle);
    float sinAngle = sin(angle);
    float2x2 rotationMatrix = float2x2(cosAngle, -sinAngle, sinAngle, cosAngle);
    return rotationMatrix * uv;
}

half4 main(float2 fragCoord) {
    // Get background color
    half4 baseColor = composable.eval(fragCoord);
    
    float speed = 2.0;

    float2 uv = fragCoord.xy / size.xy;
    uv.x *= (size.x / size.y);
    uv = rotate(uv, rotationAngle);

    float2 GA;
    GA.x = 0.0;
    GA.y -= time * 1.25;
    GA *= speed;

    float F1, F2, F3, F4, F5, N1, N2, N3, N4, N5;

    F1 = 1.0 - cellular2x2((uv + (GA * 0.1)) * 8.0);
    N1 = smoothstep(0.998, 1.0, F1);

    F2 = 1.0 - cellular2x2((uv + (GA * 0.2)) * 6.0);
    N2 = smoothstep(0.995, 1.0, F2) * 0.85;

    F3 = 1.0 - cellular2x2((uv + (GA * 0.4)) * 4.0);
    N3 = smoothstep(0.99, 1.0, F3) * 0.65;

    F4 = 1.0 - cellular2x2((uv + (GA * 0.6)) * 3.0);
    N4 = smoothstep(0.98, 1.0, F4) * 0.4;

    F5 = 1.0 - cellular2x2((uv + GA) * 1.2);
    N5 = smoothstep(0.98, 1.0, F5) * 0.25;

    float Snowout = N1 + N2 + N3 + N4 + N5;
    Snowout = clamp(Snowout, 0.0, 1.0);

    // Mix the snowflake effect smoothly into the background
    return half4(
        mix(baseColor.r, 1.0, Snowout),
        mix(baseColor.g, 1.0, Snowout),
        mix(baseColor.b, 1.0, Snowout),
        baseColor.a
    );
}
"""

val rippleShader = """
    uniform float2 size;
    uniform float time;
    uniform shader composable;
    
    half4 main(float2 fragCoord) {
        float scale = 1 / size.x;
        float2 scaledCoord = fragCoord * scale;
        float2 center = size * 0.5 * scale;
        float dist = distance(scaledCoord, center);
        float2 dir = scaledCoord - center;
        float sin = sin(dist * 70 - time * 6.28);
        float2 offset = dir * sin;
        float2 textCoord = scaledCoord + offset / 30;
        return composable.eval(textCoord / scale);
    }
""".trimIndent()

//val lightBallShader = """
//    uniform float2 size;
//    uniform float time;
//
//    vec4 main(vec2 fragCoord) {
//        float4 o = float4(0);
//        float2 p = float2(0), c = p, u = fragCoord * 2.0 - size.xy;
//        float a;
//        for (float i = 0.0; i < 4e2; i++) {
//            a = i / 2e2 - 1.0;
//            p = cos(i * 2.4 + time + float2(0.0, 11.0)) * sqrt(1.0 - a * a);
//            c = u / size.y + float2(p.x, a) / (p.y + 2.0);
//            o += (cos(i + float4(0.0, 2.0, 4.0, 0.0)) + 1.0) / dot(c, c) * (1.0 - p.y) / 3e4;
//        }
//    return o;
//}
//""".trimIndent()

//val lightBallShader = """
//    uniform float2 size;
//    uniform float time;
//    uniform shader composable;
//
//    half4 main(float2 fragCoord) {
//    float4 o = float4(0.0);
//    float2 p = float2(0.0);
//    float2 c = float2(0.0);
//    float2 u = fragCoord.xy * 2.0 - size.xy;
//
//    for (float i = 0.0; i < 180.0; i++) {
//        float a = i / 90.0 - 1.0;
//        p = cos(i * 2.4 + time + float2(0.0, 11.0)) * sqrt(1.0 - a * a);
//        c = u / size.y + float2(p.x, a) / (p.y + 2.0);
//        o += (cos(i + float4(0.0, 2.0, 4.0, 0.0)) + 1.0) / dot(c, c) * (1.0 - p.y) / 30000.0;
//    }
//
//    return half4(o.rgb, 1.0);
//}
//""".trimIndent()

val lightBallShader = """
    uniform float2 size;
    uniform float time;
    uniform shader composable;

    half4 main(float2 fragCoord) {
    float4 o = float4(0.0);
    float2 u = fragCoord.xy * 2.0 - size.xy;
    float2 s = u / size.y;

    for (float i = 0.0; i < 180.0; i++) {
        float a = i / 90.0 - 1.0;
        float sqrtTerm = sqrt(1.0 - a * a);
        float2 p = cos(i * 2.4 + time + float2(0.0, 11.0)) * sqrtTerm;
        float2 c = s + float2(p.x, a) / (p.y + 2.0);
        float denom = dot(c, c);
        float4 cosTerm = cos(i + float4(0.0, 2.0, 4.0, 0.0)) + 1.0;
        o += cosTerm / denom * (1.0 - p.y) / 30000.0;
    }

    return half4(o.rgb, 1.0);
}
""".trimIndent()

val seaShader = """
/* 
 * "Seascape" Shader in AGSL
 * Original GLSL code by Alexander Alekseev aka TDM - 2014
 * Converted to AGSL and modified
 * License: Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License.
 * Contact: tdmaav@gmail.com
 */

uniform float time;
uniform float2 size;

const int NUM_STEPS = 32;
const float PI = 3.141592;
const float EPSILON = 1e-3;
const float EPSILON_NRM = (0.1 / 1000.0);

const int ITER_GEOMETRY = 3;
const int ITER_FRAGMENT = 5;
const float SEA_HEIGHT = 0.6;
const float SEA_CHOPPY = 4.0;
const float SEA_SPEED = 0.8;
const float SEA_FREQ = 0.16;
const float3 SEA_BASE = float3(0.0, 0.09, 0.18);
const float3 SEA_WATER_COLOR = float3(0.8, 0.9, 0.6) * 0.6;
const float2x2 octave_m = float2x2(
    float2(1.6, -1.2),
    float2(1.2, 1.6)
);

// Function to create a rotation matrix from Euler angles
float3x3 fromEuler(float3 ang) {
    float2 a1 = float2(sin(ang.x), cos(ang.x));
    float2 a2 = float2(sin(ang.y), cos(ang.y));
    float2 a3 = float2(sin(ang.z), cos(ang.z));
    float3x3 m;
    m[0] = float3(
        a1.y * a3.y + a1.x * a2.x * a3.x,
        a1.y * a2.x * a3.x + a3.y * a1.x,
        -a2.y * a3.x
    );
    m[1] = float3(
        -a2.y * a1.x,
        a1.y * a2.y,
        a2.x
    );
    m[2] = float3(
        a3.y * a1.x * a2.x + a1.y * a3.x,
        a1.x * a3.x - a1.y * a3.y * a2.x,
        a2.y * a3.y
    );
    return m;
}

// Hash function
float hash(float2 p) {
    float h = dot(p, float2(127.1, 311.7));
    return fract(sin(h) * 43758.5453123);
}

// Noise function
float noise(float2 p) {
    float2 i = floor(p);
    float2 f = fract(p);
    float2 u = f * f * (3.0 - 2.0 * f);
    return -1.0 + 2.0 * mix(
        mix(hash(i + float2(0.0, 0.0)), hash(i + float2(1.0, 0.0)), u.x),
        mix(hash(i + float2(0.0, 1.0)), hash(i + float2(1.0, 1.0)), u.x),
        u.y
    );
}

// Diffuse lighting
float diffuse(float3 n, float3 l, float p) {
    return pow(dot(n, l) * 0.4 + 0.6, p);
}

// Specular lighting
float specular(float3 n, float3 l, float3 e, float s) {
    float nrm = (s + 8.0) / (PI * 8.0);
    return pow(max(dot(reflect(e, n), l), 0.0), s) * nrm;
}

// Sky color
float3 getSkyColor(float3 e) {
    e.y = (max(e.y, 0.0) * 0.8 + 0.2) * 0.8;
    return float3(
        pow(1.0 - e.y, 2.0),
        1.0 - e.y,
        0.6 + (1.0 - e.y) * 0.4
    ) * 1.1;
}

// Sea octave function
float sea_octave(float2 uv, float choppy) {
    uv += noise(uv);
    float2 wv = 1.0 - abs(sin(uv));
    float2 swv = abs(cos(uv));
    wv = mix(wv, swv, wv);
    return pow(1.0 - pow(wv.x * wv.y, 0.65), choppy);
}

// Map function for geometry iterations
float map(float3 p) {
    float SEA_TIME = 1.0 + time * SEA_SPEED;
    float freq = SEA_FREQ;
    float amp = SEA_HEIGHT;
    float choppy = SEA_CHOPPY;
    float2 uv = float2(p.x, p.z);
    uv.x *= 0.75;

    float d, h = 0.0;
    for (int i = 0; i < ITER_GEOMETRY; i++) {
        d = sea_octave((uv + SEA_TIME) * freq, choppy);
        d += sea_octave((uv - SEA_TIME) * freq, choppy);
        h += d * amp;
        uv = octave_m * uv;
        freq *= 1.9;
        amp *= 0.22;
        choppy = mix(choppy, 1.0, 0.2);
    }
    return p.y - h;
}

// Map function for fragment iterations
float map_detailed(float3 p) {
    float SEA_TIME = 1.0 + time * SEA_SPEED;
    float freq = SEA_FREQ;
    float amp = SEA_HEIGHT;
    float choppy = SEA_CHOPPY;
    float2 uv = float2(p.x, p.z);
    uv.x *= 0.75;

    float d, h = 0.0;
    for (int i = 0; i < ITER_FRAGMENT; i++) {
        d = sea_octave((uv + SEA_TIME) * freq, choppy);
        d += sea_octave((uv - SEA_TIME) * freq, choppy);
        h += d * amp;
        uv = octave_m * uv;
        freq *= 1.9;
        amp *= 0.22;
        choppy = mix(choppy, 1.0, 0.2);
    }
    return p.y - h;
}

// Normal calculation
float3 getNormal(float3 p, float eps) {
    float3 n;
    n.y = map_detailed(p);
    n.x = map_detailed(float3(p.x + eps, p.y, p.z)) - n.y;
    n.z = map_detailed(float3(p.x, p.y, p.z + eps)) - n.y;
    n.y = eps;
    return normalize(n);
}

// Sea color calculation
float3 getSeaColor(float3 p, float3 n, float3 l, float3 eye, float3 dist) {
    float fresnel = clamp(1.0 - dot(n, -eye), 0.0, 1.0);
    fresnel = min(pow(fresnel, 3.0), 0.5);

    float3 reflected = getSkyColor(reflect(eye, n));
    float3 refracted = SEA_BASE + diffuse(n, l, 80.0) * SEA_WATER_COLOR * 0.12;

    float3 color = mix(refracted, reflected, fresnel);

    float atten = max(1.0 - dot(dist, dist) * 0.001, 0.0);
    color += SEA_WATER_COLOR * (p.y - SEA_HEIGHT) * 0.18 * atten;

    color += specular(n, l, eye, 60.0);

    return color;
}

// Height map tracing result struct
struct HeightMapTracingResult {
    float t;
    float3 p;
};

// Height map tracing function
HeightMapTracingResult heightMapTracing(float3 ori, float3 dir) {
    float tm = 0.0;
    float tx = 1000.0;
    float hx = map(ori + dir * tx);
    float3 p = ori + dir * tx;
    if (hx > 0.0) {
        return HeightMapTracingResult(tx, p);
    }
    float hm = map(ori);
    for (int i = 0; i < NUM_STEPS; i++) {
        float tmid = mix(tm, tx, hm / (hm - hx));
        p = ori + dir * tmid;
        float hmid = map(p);
        if (hmid < 0.0) {
            tx = tmid;
            hx = hmid;
        } else {
            tm = tmid;
            hm = hmid;
        }
        if (abs(hmid) < EPSILON) break;
    }
    float t = mix(tm, tx, hm / (hm - hx));
    p = ori + dir * t;
    return HeightMapTracingResult(t, p);
}

// Pixel color calculation
float3 getPixel(float2 coord) {
    float2 uv = coord / size;
    uv = uv * 2.0 - 1.0;
    uv.x *= size.x / size.y;

    // Ray
    float3 ang = float3(
        sin(time * 3.0) * 0.1,
        sin(time) * 0.2 + 0.3,
        time
    );
    float3 ori = float3(0.0, 3.5, time * 5.0);
    float3 dir = normalize(float3(uv.xy, -2.0));
    dir.z += length(uv) * 0.14;
    dir = fromEuler(ang) * dir;

    // Tracing
    HeightMapTracingResult hmtResult = heightMapTracing(ori, dir);
    float3 p = hmtResult.p;
    float3 dist = p - ori;
    float3 n = getNormal(p, dot(dist, dist) * EPSILON_NRM);
    float3 light = normalize(float3(0.0, 1.0, 0.8));

    // Color
    float3 color = mix(
        getSkyColor(dir),
        getSeaColor(p, n, light, dir, dist),
        pow(smoothstep(0.0, -0.02, dir.y), 0.2)
    );

    return color;
}

// Main function
half4 main(float2 fragCoord) {
    float3 color = getPixel(fragCoord);

    // Post-processing
    color = pow(color, float3(0.65));

    return half4(color, 1.0);
}
""".trimIndent()


private const val FRACTAL_SHADER_SRC = """
    uniform float2 size;
    uniform float time;
    uniform shader composable;
    
    float f(float3 p) {
        p.z -= time * 5.;
        float a = p.z * .1;
        p.xy *= mat2(cos(a), sin(a), -sin(a), cos(a));
        return .1 - length(cos(p.xy) + sin(p.yz));
    }
    
    half4 main(float2 fragcoord) { 
        float3 d = .5 - fragcoord.xy1 / size.y;
        float3 p=float3(0);
        for (int i = 0; i < 32; i++) {
          p += f(p) * d;
        }
        return ((sin(p) + float3(2, 5, 12)) / length(p)).xyz1;
    }
"""

private const val CLOUD_SHADER_SRC = """
uniform float2 size;
uniform float time;
uniform shader composable;
const float cloudscale = 1.1;
const float speed = 0.03;
const float clouddark = 0.5;
const float cloudlight = 0.3;
const float cloudcover = 0.2;
const float cloudalpha = 8.0;
const float skytint = 0.5;
const vec3 skycolour1 = vec3(0.2, 0.4, 0.6);
const vec3 skycolour2 = vec3(0.4, 0.7, 1.0);

const mat2 m = mat2( 1.6,  1.2, -1.2,  1.6 );

vec2 hash( vec2 p ) {
	p = vec2(dot(p,vec2(127.1,311.7)), dot(p,vec2(269.5,183.3)));
	return -1.0 + 2.0*fract(sin(p)*43758.5453123);
}

float noise( in vec2 p ) {
    const float K1 = 0.366025404; // (sqrt(3)-1)/2;
    const float K2 = 0.211324865; // (3-sqrt(3))/6;
	vec2 i = floor(p + (p.x+p.y)*K1);	
    vec2 a = p - i + (i.x+i.y)*K2;
    vec2 o = (a.x>a.y) ? vec2(1.0,0.0) : vec2(0.0,1.0); //vec2 of = 0.5 + 0.5*vec2(sign(a.x-a.y), sign(a.y-a.x));
    vec2 b = a - o + K2;
	vec2 c = a - 1.0 + 2.0*K2;
    vec3 h = max(0.5-vec3(dot(a,a), dot(b,b), dot(c,c) ), 0.0 );
	vec3 n = h*h*h*h*vec3( dot(a,hash(i+0.0)), dot(b,hash(i+o)), dot(c,hash(i+1.0)));
    return dot(n, vec3(70.0));	
}

float fbm(vec2 n) {
	float total = 0.0, amplitude = 0.1;
	for (int i = 0; i < 7; i++) {
		total += noise(n) * amplitude;
		n = m * n;
		amplitude *= 0.4;
	}
	return total;
}

half4 main(in vec2 fragCoord ) {
    vec2 p = fragCoord.xy / size.xy;
	vec2 uv = p*vec2(size.x/size.y,1.0);    
    float time = time * speed;
    float q = fbm(uv * cloudscale * 0.5);
    
    //ridged noise shape
	float r = 0.0;
	uv *= cloudscale;
    uv -= q - time;
    float weight = 0.8;
    for (int i=0; i<8; i++){
		r += abs(weight*noise( uv ));
        uv = m*uv + time;
		weight *= 0.7;
    }
    
    //noise shape
	float f = 0.0;
    uv = p*vec2(size.x/size.y,1.0);
	uv *= cloudscale;
    uv -= q - time;
    weight = 0.7;
    for (int i=0; i<8; i++){
		f += weight*noise( uv );
        uv = m*uv + time;
		weight *= 0.6;
    }
    
    f *= r + f;
    
    //noise colour
    float c = 0.0;
    time = time * speed * 2.0;
    uv = p*vec2(size.x/size.y,1.0);
	uv *= cloudscale*2.0;
    uv -= q - time;
    weight = 0.4;
    for (int i=0; i<7; i++){
		c += weight*noise( uv );
        uv = m*uv + time;
		weight *= 0.6;
    }
    
    //noise ridge colour
    float c1 = 0.0;
    time = time * speed * 3.0;
    uv = p*vec2(size.x/size.y,1.0);
	uv *= cloudscale*3.0;
    uv -= q - time;
    weight = 0.4;
    for (int i=0; i<7; i++){
		c1 += abs(weight*noise( uv ));
        uv = m*uv + time;
		weight *= 0.6;
    }
	
    c += c1;
    
    vec3 skycolour = mix(skycolour2, skycolour1, p.y);
    vec3 cloudcolour = vec3(1.1, 1.1, 0.9) * clamp((clouddark + cloudlight*c), 0.0, 1.0);
   
    f = cloudcover + cloudalpha*f*r;
    
    vec3 result = mix(skycolour, clamp(skytint * skycolour + cloudcolour, 0.0, 1.0), clamp(f + c, 0.0, 1.0));
    
	return vec4(result, 1.0 );
}
"""


class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        WindowCompat.setDecorFitsSystemWindows(window, false)

        setContent {
//            val  timeMs = produceDrawLoopCounter(1.0f)
            AGSLShadersTheme {
                // A surface container using the 'background' color from the theme
                LoadingView()
            }
        }
    }
}

@Composable
fun ImageView() {
//        val shader = RuntimeShader(SNOW_SHADER_SRC)
//        val shader = RuntimeShader(rippleShader)
//        val shader = RuntimeShader(FRACTAL_SHADER_SRC) // TODO: uncomment to see 2nd shader
    val shader = RuntimeShader(lightBallShader)
//        val shader = RuntimeShader(seaShader)
//        val shader = RuntimeShader(CLOUD_SHADER_SRC) // TODO: uncomment to see 3rd shader
//        val photo = BitmapFactory.decodeResource(resources, R.drawable.map)
//        val photo = BitmapFactory.decodeResource(resources, R.drawable.butterfly)
    val photo = BitmapFactory.decodeResource(LocalContext.current.resources, R.drawable.moon)
    val scope = rememberCoroutineScope()
    val timeMs = remember { mutableStateOf(0f) }
    LaunchedEffect(Unit) {
        scope.launch {
            while (true) {
                timeMs.value = (System.currentTimeMillis() % 100_000L) / 1_000f
                delay(16)
            }
        }
    }
    Box(
        modifier = Modifier
            .fillMaxSize()
    ) {
        val scope = rememberCoroutineScope()
        val timeMs = remember { mutableStateOf(0f) }
        LaunchedEffect(Unit) {
            scope.launch {
                while (true) {
                    timeMs.value = (System.currentTimeMillis() % 100_000L) / 1_000f
                    delay(16)
                }
            }
        }
        Surface(
            modifier = Modifier
                .fillMaxSize(),
            color = MaterialTheme.colorScheme.background
        ) {
            Image(
                bitmap = photo.asImageBitmap(),
                modifier = Modifier
                    .onSizeChanged { size ->
                        shader.setFloatUniform(
                            "size",
                            size.width.toFloat(),
                            size.height.toFloat()
                        )
                    }
                    .graphicsLayer {
                        clip = true
                        shader.setFloatUniform("time", timeMs.value)
                        renderEffect =
                            RenderEffect
                                .createRuntimeShaderEffect(shader, "composable")
                                .asComposeRenderEffect()
                    },
                contentScale = ContentScale.FillHeight,
                contentDescription = null,
            )
        }
        Text(
            text = "@Techblog_AGSL",
            modifier = Modifier
                .align(Alignment.TopCenter)
                .offset(y = (LocalConfiguration.current.screenHeightDp * 0.975f).dp),
            color = Color.LightGray,
            textAlign = TextAlign.Center,
            fontSize = 10.sp
        )
    }
}

@Composable
fun LoadingView() {
    val scope = rememberCoroutineScope()
    val timeMs = remember { mutableStateOf(0f) }
    LaunchedEffect(Unit) {
        scope.launch {
            while (true) {
                timeMs.value = (System.currentTimeMillis() % 100_000L) / 1_000f
                delay(16)
            }
        }
    }
    val shader = RuntimeShader(lightBallShader)
    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(Color.Black)
    ) {
        Surface(
            modifier = Modifier
                .width(360.dp)
                .height(360.dp)
                .align(Alignment.TopCenter)
                .offset(y = (LocalConfiguration.current.screenHeightDp * 0.2f).dp),
            color = MaterialTheme.colorScheme.background
        ) {
            Box(
                modifier = Modifier
                    .onSizeChanged { size ->
                        Log.e("onSizeChanged", "size: $size")
                        shader.setFloatUniform(
                            "size",
                            size.width.toFloat(),
                            size.height.toFloat()
                        )
                    }
                    .graphicsLayer {
                        with(shader) {
                            setFloatUniform("time", timeMs.value)
                        }
                        renderEffect =
                            RenderEffect
                                .createRuntimeShaderEffect(shader, "composable")
                                .asComposeRenderEffect()
                    }
                    .fillMaxSize()
                    .background(Color.Black)
            )
        }

        AnimatedLoadingText(
            modifier = Modifier
                .wrapContentSize()
                .background(Color.Transparent)
                .align(Alignment.TopCenter)
                .offset(y = (LocalConfiguration.current.screenHeightDp * 0.7f).dp)
        )

        Text(
            text = "@Techblog_AGSL",
            modifier = Modifier
                .align(Alignment.TopCenter)
                .offset(y = (LocalConfiguration.current.screenHeightDp * 0.975f).dp),
            color = Color.LightGray,
            textAlign = TextAlign.Center,
            fontSize = 10.sp
        )
    }
}

@Composable
fun AnimatedLoadingText(modifier: Modifier) {
    var loadingText by remember {
        mutableStateOf("Loading")
    }

    LaunchedEffect(Unit) {
        val loadingStates = listOf("Loading", "Loading.", "Loading..", "Loading...")
        var index = 0
        while (true) {
            loadingText = loadingStates[index]
            index = (index + 1) % loadingStates.size
            delay(500L)
        }
    }

    Box(
        modifier = modifier,
    ) {
        Text(
            modifier = Modifier.wrapContentSize(),
            text = loadingText,
            style = TextStyle.Default.copy(
                fontSize = 24.sp,
            ),
            color = Color.LightGray
        )
    }
}

@Composable
fun produceDrawLoopCounter(speed: Float = 1f): State<Float> {
    return produceState(0f) {
        while (true) {
            withInfiniteAnimationFrameMillis {
                value = it / 1000f * speed
            }
        }
    }
}


