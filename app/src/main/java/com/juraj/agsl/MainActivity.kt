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


class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        WindowCompat.setDecorFitsSystemWindows(window, false)

        setContent {
            AGSLShadersTheme {
                // A surface container using the 'background' color from the theme
                ImageView()
//                LoadingView()
            }
        }
    }
}

@Composable
fun ImageView() {
    // TODO: uncomment to seethe shader you want
    val shader = RuntimeShader(snowShader)
//    val shader = RuntimeShader(rippleShader)
//    val shader = RuntimeShader(fractalShader)
//    val shader = RuntimeShader(lightBallShader)
//    val shader = RuntimeShader(cloudShader)

//    val photo = BitmapFactory.decodeResource(resources, R.drawable.map)
//    val photo = BitmapFactory.decodeResource(resources, R.drawable.butterfly)
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
        modifier = Modifier.fillMaxSize()
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
            modifier = Modifier.fillMaxSize(), color = MaterialTheme.colorScheme.background
        ) {
            Image(
                bitmap = photo.asImageBitmap(),
                modifier = Modifier
                    .onSizeChanged { size ->
                        shader.setFloatUniform(
                            "size", size.width.toFloat(), size.height.toFloat()
                        )
                    }
                    .graphicsLayer {
                        clip = true
                        shader.setFloatUniform("time", timeMs.value)
                        renderEffect = RenderEffect
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
            Box(modifier = Modifier
                .onSizeChanged { size ->
                    Log.e("onSizeChanged", "size: $size")
                    shader.setFloatUniform(
                        "size", size.width.toFloat(), size.height.toFloat()
                    )
                }
                .graphicsLayer {
                    with(shader) {
                        setFloatUniform("time", timeMs.value)
                    }
                    renderEffect = RenderEffect
                        .createRuntimeShaderEffect(shader, "composable")
                        .asComposeRenderEffect()
                }
                .fillMaxSize()
                .background(Color.Black))
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


