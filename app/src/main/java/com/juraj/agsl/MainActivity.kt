package com.juraj.agsl

import android.graphics.BitmapFactory
import android.graphics.RenderEffect
import android.graphics.RuntimeShader
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.offset
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asComposeRenderEffect
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.layout.onSizeChanged
import androidx.compose.ui.platform.LocalConfiguration
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

//        val shader = RuntimeShader(SNOW_SHADER_SRC)
        val shader = RuntimeShader(rippleShader)
//        val shader = RuntimeShader(FRACTAL_SHADER_SRC) // TODO: uncomment to see 2nd shader
//        val shader = RuntimeShader(CLOUD_SHADER_SRC) // TODO: uncomment to see 3rd shader
//        val photo = BitmapFactory.decodeResource(resources, R.drawable.map)
//        val photo = BitmapFactory.decodeResource(resources, R.drawable.butterfly)
        val photo = BitmapFactory.decodeResource(resources, R.drawable.moon)

        setContent {
            val scope = rememberCoroutineScope()
            val timeMs = remember { mutableStateOf(0f) }
            LaunchedEffect(Unit) {
                scope.launch {
                    while (true) {
                        timeMs.value = (System.currentTimeMillis() % 100_000L) / 1_000f
                        delay(10)
                    }
                }
            }

            AGSLShadersTheme {
                // A surface container using the 'background' color from the theme
                Box(
                    modifier = Modifier
                        .fillMaxSize()
                ) {
                    Surface(
                        modifier = Modifier
//                        .aspectRatio(1f)
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
        }
    }
}
