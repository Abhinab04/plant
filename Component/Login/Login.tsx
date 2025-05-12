import React, { useState } from "react";
import { View, Text, StyleSheet, Pressable, TextInput, Image } from "react-native"

const login = () => {

    const [signup, setsignup] = useState(false)

    return (
        <View style={styles.container}>
            <View>
                <Text style={styles.title}>Sproutelle</Text>
            </View>
            <View style={styles.buttonContainer}>
                <Pressable onPress={() => setsignup(true)}
                    style={[signup && styles.active]}
                >
                    <Text style={signup ? styles.actives : styles.button}>SignUp</Text>
                </Pressable>
                <Pressable onPress={() => setsignup(false)}
                    style={[!signup && styles.active]}
                >
                    <Text style={!signup ? styles.actives : styles.button}>Login</Text>
                </Pressable>
            </View>
            <View style={styles.in}>
                {signup && (
                    <TextInput
                        placeholder="Username"
                        placeholderTextColor="#F6FCDF"
                        style={styles.input}
                    />
                )}

                <TextInput
                    placeholder="Email"
                    placeholderTextColor="#F6FCDF"
                    style={styles.input}
                />
                <TextInput
                    placeholder="password"
                    placeholderTextColor="#F6FCDF"
                    style={styles.input}
                    secureTextEntry={true}
                />
            </View>
            <View style={styles.buttonContainers}>
                <Pressable style={[styles.socialbutton, styles.google]}>
                    <View style={{ flexDirection: 'row', alignItems: 'center', gap: 7 }}>
                        <Text style={styles.add}>Continue with Google</Text>
                    </View>
                </Pressable>
                <Pressable style={[styles.socialbutton, styles.gmailButton]}>
                    <Text style={styles.add}>Continue with Gmail</Text>
                </Pressable>
                <Pressable style={[styles.socialbutton, styles.facebookButton]}>
                    <Text style={styles.add}>Continue with Facebook</Text>
                </Pressable>
            </View>
        </View>
    )
}

const styles = StyleSheet.create({
    title: {
        color: "#F6FCDF",
        fontSize: 47,
        fontWeight: 'bold',
        marginBottom: "10%",
        marginTop: "10%"
    },
    container: {
        flex: 1, // makes it take full screen height
        justifyContent: "center",
        alignItems: 'center',
        padding: 20,
        backgroundColor: "#1F1F1F",
        gap: 20,
    },
    buttonContainer: {
        flexDirection: 'row',
        marginTop: "5%",
        gap: 30,
    },
    button: {
        borderWidth: 2,
        borderColor: '#91AC8F',
        backgroundColor: "#91AC8F",
        color: '#1A1A19',
        paddingHorizontal: 50,
        paddingVertical: 15,
        fontSize: 18,
        fontWeight: "800",
        borderRadius: 50
    },
    input: {
        backgroundColor: "#1E1E1E",
        borderColor: "#91AC8F",
        borderWidth: 1,
        borderRadius: 25,
        paddingVertical: 15,
        paddingHorizontal: 20,
        fontSize: 16,
        color: "#F6FCDF",
    },
    in: {
        gap: 23,
        marginTop: 30,
        width: "90%"
    },
    add: {
        paddingVertical: 14,
        fontSize: 19,
        fontWeight: 800,
        color: '#1A1A19',

    },
    buttonContainers: {
        marginTop: 30,
        gap: 15,
        width: '90%',

    },
    socialbutton: {
        borderRadius: 50,
        paddingVertical: 10,
        marginBottom: 10,
        alignItems: "center",
    },
    google: {
        backgroundColor: "#91AC8F",
    },
    gmailButton: {
        backgroundColor: "#91AC8F",
    },
    facebookButton: {
        backgroundColor: "#91AC8F",
    },
    actives: {
        borderWidth: 2,
        borderColor: '#91AC8F',
        backgroundColor: "#304D30",
        color: '#B6C4B6',
        paddingHorizontal: 50,
        paddingVertical: 15,
        fontSize: 18,
        fontWeight: "800",
        borderRadius: 50
    },
    active: {
        borderWidth: 1,
        borderRadius: 50,
    }
})

export default login;